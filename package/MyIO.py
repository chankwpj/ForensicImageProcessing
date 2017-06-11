import cv2

def ImageWrite(image, prefixName, fileFormat, mask=None):
    
    #write mask
    out = cv2.bitwise_and(image,image, mask=mask)
    fileName = str(prefixName) + str(fileFormat)
    cv2.imwrite(fileName, out)

    #also write the mask
    if mask is not None :
        fileName = str(prefixName) + "Mask" +  str(fileFormat)
        cv2.imwrite(fileName, mask)