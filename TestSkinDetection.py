import glob
import numpy as np
import cv2
import MyIO as IO
from SkinDetection import SkinDetection

'''
imageFolderPath = './dataset2'
imagePath = glob.glob(imageFolderPath +'/*.jpg') 
imagePath = imagePath + glob.glob(imageFolderPath +'/*.png') 
imageFolderPath = './general-bruise'
imagePath = imagePath + glob.glob(imageFolderPath +'/*.jpg') 
'''
imageFolderPath = './DataSet'
imagePath = glob.glob(imageFolderPath +'/*.jpg') 
skinDetection = SkinDetection();

for index, path in enumerate(imagePath):
    imgName = "Img" + str(index) 
    im = cv2.imread(path)
    #mask, m1, m2 = skinDetection.SkinThresholding(im, True, True)
    mask = skinDetection.mask(im)
    IO.ImageWrite(im, imgName, ".jpg", mask=mask)
