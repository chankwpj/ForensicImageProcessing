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

#select = ["Image-48-clean", "Image-201-clean", "Image-208-clean", "Image-88-clean", "292497", "294828", "Image-19-clean", "Image-30-clean", "Image-182-clean"]
select = [ "Image-115-clean"]

for index, path in enumerate(imagePath):
    temp = path.split('\\')
    imgName = temp[len(temp)-1].split('.')[0]
    '''
    b = -1
    for index, s in enumerate(select):
        if (imgName == s):
            b = 1
            break
    if (b<0):
        continue
    '''

    #imgName = "Img" + str(index) 
    print "Working on Img " + imgName 
    im = cv2.imread(path)
    if (index < len(imagePathBlue)):
        #mask, mmUnit = myPreProcessor1.mask(im)
        preImg, mmUnit = myPreProcessor1.process(im)
        windowMask, rulerMask, erodedMask, Tpoints, circles, unit, pt = myBruiseWindow1.getWindowMask(im)
    else:
        #mask, mmUnit = myPreProcessor2.mask(im)
        preImg, mmUnit = myPreProcessor2.process(im)
        windowMask, rulerMask, erodedMask, Tpoints, circles, unit, pt = myBruiseWindow2.getWindowMask(im)
    
    #cv2.imwrite(imgName+"Pre.jpg", preImg)
    #[resKMeans, resLabels] = myBruiseSegmentation.kMeansGeneration(preIm)
    #cv2.imwrite(imgName + "kMeans.jpg", resKMeans)
    mask = myBruiseSegmentation.mask(preImg, imgName)

    #measurement 
    [box, rect] = measurmentTool.measure(mask)
    cv2.drawContours(im,[box],0,(255,0,0),5)
    cv2.putText(im, str(int(rect[1][0]*mmUnit))+"x"+str(int(rect[1][1]*mmUnit)), (500,1200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0),2)

    cv2.imwrite(imgName + "mask.jpg", mask)

    rgbMask = cv2.cvtColor(erodedMask, cv2.COLOR_GRAY2BGR)
    ##mark circles etc to out
    #draw all circles
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(im,(i[0],i[1]),i[2],(255,0,0),2)
        cv2.circle(rgbMask,(i[0],i[1]),i[2],(255,0,0),2)
    #draw target points
    for tp in enumerate(Tpoints):
        cv2.circle(im, tp[1],25,(0,0,255),-1)
        cv2.circle(rgbMask, tp[1],25,(0,0,255),-1)

    #indicate which circle used as convertor
    cv2.circle(im,(pt[0],pt[1]),25,(0,255,255),-1)
    cv2.circle(rgbMask,(pt[0],pt[1]),25,(0,255,255),-1)


    cv2.imwrite(imgName + "out.jpg", im)
    cv2.imwrite(imgName + "rulerMask.jpg", rgbMask)
   

