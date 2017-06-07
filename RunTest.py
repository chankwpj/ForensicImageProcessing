import glob
import numpy as np
import cv2
import MyIO as IO
from BruiseWindow import *
from SkinDetection import *
from NoiseRemoval import *
from PreProcessor import *
from BruiseSegmentation import *

imageFolderPath = './D2Blue'
imagePathBlue = glob.glob(imageFolderPath +'/*.jpg') 
imagePathBlue= imagePathBlue + glob.glob(imageFolderPath +'/*.png') 

imageFolderPath = './D2BlueX'
imagePathBlueX = glob.glob(imageFolderPath +'/*.jpg') 
imagePathBlueX= imagePathBlueX + glob.glob(imageFolderPath +'/*.png') 

imagePath = imagePathBlue + imagePathBlueX


mySkinDetector1 = SkinDetection()
myBruiseWindow1 = BruiseWindow()
myPreProcessor1 = PreProcessor([mySkinDetector1, myBruiseWindow1])

myBruiseWindow2 = BruiseWindow()
myBruiseWindow2.setNoiseRemoval(None);
myPreProcessor2 = PreProcessor([mySkinDetector1, myBruiseWindow2])

myBruiseSegmentation1 = BruiseSegmentation(myPreProcessor1)
myBruiseSegmentation2 = BruiseSegmentation(myPreProcessor2)


for index, path in enumerate(imagePath):
    imgName = "Img" + str(index) 
    print "Working on Img " + imgName 
    im = cv2.imread(path)
    if (index < len(imagePathBlue)):
        mask = myBruiseSegmentation1.mask(im)
    else:
        mask = myBruiseSegmentation2.mask(im)

    #measurement 

    #cv2.imwrite(imgName + "out.jpg", mask)
    IO.ImageWrite(im, imgName + "out", '.jpg', mask)


