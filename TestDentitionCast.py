import numpy as np
import cv2
import string
import os
import os.path
import DentitionCast as cast
import MyIO as IO

paths = [];
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        p = os.path.join(dirpath, filename)
        if ("control-set" in p) :
            continue
        if ("Study-models" in p) :
            paths.append(p)
          

#Parameters for RemoveRuler
Rlower = np.array([150, 150, 150]) # okay
Rupper = np.array([255, 255, 255])  
hShift = -20
vShift = 90
#Parameters for LocalizeCast
Clower = 100
Cupper = 255
dilationKsize = 11
dilationIteration = 4

for index in range(len(paths)):
    if (index % 3 != 0):
        print("Extracting the cast from: " +  paths[index])
        im = cv2.imread(paths[index]);
        mask = cast.RemoveRuler(im, Rlower, Rupper, hShift, vShift)
        im = cv2.bitwise_and(im,im,mask=mask)
        mask = cast.LocalizeCast(im, Clower, Cupper, dilationKsize, dilationIteration)
        im = cv2.bitwise_and(im,im,mask=mask)
        #cv2.imwrite(str(index) + "cast.jpg", im)
        IO.ImageWrite(im, "Cast" + str(index), ".jpg")