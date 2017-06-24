import glob
import numpy as np
import cv2
import MyIO as IO
from BruiseWindow import *
from SkinDetection import *
from NoiseRemoval import *
from PreProcessor import *
from BruiseSegmentation import *
from MeasurmentTool import *

imageFolderPath = './D2Blue'
imagePathBlue = glob.glob(imageFolderPath +'/*.jpg') 
imagePathBlue= imagePathBlue + glob.glob(imageFolderPath +'/*.png') 

imageFolderPath = './D2BlueX'
imagePathBlueX = glob.glob(imageFolderPath +'/*.jpg') 
imagePathBlueX= imagePathBlueX + glob.glob(imageFolderPath +'/*.png') 

imagePath = imagePathBlue + imagePathBlueX


mySkinDetector1 = SkinDetection()
myBruiseWindow1 = BruiseWindow(NoiseRemoval())
myPreProcessor1 = PreProcessor(mySkinDetector1, myBruiseWindow1)

myBruiseWindow2 = BruiseWindow(None)
myPreProcessor2 = PreProcessor(mySkinDetector1, myBruiseWindow2)

myBruiseSegmentation = BruiseSegmentation()
measurmentTool = MeasurmentTool()

for index, path in enumerate(imagePath):
    temp = path.split('\\')
    imgName = temp[len(temp)-1].split('.')[0]
    #imgName = "Img" + str(index) 
    print "Working on Img " + imgName 
    im = cv2.imread(path)
    if (index < len(imagePathBlue)):
        windowMask, rulerMask, erodedMask, Tpoints, circles, unit, pt = myBruiseWindow1.getWindowMask(im)
        #preIm, mmUnit = myPreProcessor1.process(im)
    else:
        windowMask, rulerMask, erodedMask, Tpoints, circles, unit, pt = myBruiseWindow2.getWindowMask(im)
        #preIm, mmUnit = myPreProcessor2.process(im)
    
    im = cv2.imread("./SegmentationResult/" + imgName + "out.jpg")
    rgbMask = cv2.cvtColor(erodedMask, cv2.COLOR_GRAY2BGR)
    
    #draw all circles
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),5)
        cv2.circle(rgbMask,(i[0],i[1]),i[2],(0,255,0),15)
        # draw the center of the circle
        #cv2.circle(im,(i[0],i[1]),4,(255,0,0),-1)
    #draw target points
    for tp in enumerate(Tpoints):
        cv2.circle(im, tp[1],25,(0,0,255),-1)
        cv2.circle(rgbMask, tp[1],25,(0,0,255),-1)

    #indicate which circle used as convertor
    cv2.circle(im,(pt[0],pt[1]),25,(0,255,255),-1)
    cv2.circle(rgbMask,(pt[0],pt[1]),25,(0,255,255),-1)
    #cv2.imwrite(imgName + str("Window") + ".jpg", im)
    cv2.imwrite(imgName + str("RulerMask") + ".jpg", rgbMask)

    '''
    getWindowMask()
    rulerMask = self.extractRuler(im)
    circles = self.getAllDetectedCircles(rulerMask)  
    pt0, pt1, pt2, pt3, gridWithNoPoint = self.selectTheCornerCircles(rulerMask, circles)
    unit = self.getRatio(circles, [pt0, pt1, pt2, pt3])
    windowMask, Tpoints = self.threePointsToWindow(np.zeros(rulerMask.shape),pt0,pt1,pt2,pt3,gridWithNoPoint)
    return [windowMask, rulerMask, erodedMask, Tpoints, circles, unit, pt];
    '''