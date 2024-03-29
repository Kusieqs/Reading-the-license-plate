import os
import cv2
import imutils as imu
import numpy as np
import easyocr

def ShowImage(name,img): #Additional method to show image (You have to remove code from 30 - 37)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def PrintingPlate(cnts, gray):
    for k in cnts:
        (x,y,w,h) = cv2.boundingRect(k)

        if w >= 60 and w <= 350:

            plate = gray[y:y + h, x:x + w]
            plate = imu.resize(plate, width=350)
            plate = cv2.medianBlur(plate, 3)
            roi = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            roi = cv2.dilate(roi, None, iterations = 2)
            roi = cv2.erode(roi, None, iterations= 3)
            roi = cv2.dilate(roi, None, iterations = 1)
            cv2.imshow('roi',roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def ErodeDilate(thresh,light):
    newthresh = cv2.bitwise_and(thresh,thresh,mask=light)
    newthresh = cv2.dilate(newthresh,None,iterations=3)
    newthresh = cv2.erode(newthresh,None,iterations=5)
    newthresh = cv2.dilate(newthresh,None,iterations=6)
    newthresh = cv2.erode(newthresh, None, iterations=4)
    newthresh = cv2.dilate(newthresh,None,iterations=2)
    return newthresh

def SetCandidates(nameOfImage, img):

    # Resizes the image to a width of 600px, Convert the resized image to grayscale, Applies a sharpening filter to the grayscale
    img = imu.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    sharp = cv2.filter2D(gray, -1 , sharpen)

    # Morphological blackhat operation
    rectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,13))
    blackHat = cv2.morphologyEx(sharp,cv2.MORPH_BLACKHAT,rectangleKernel)

    # Closing morphology operation.
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    light = cv2.morphologyEx(sharp,cv2.MORPH_CLOSE,squareKernel)

    # Thresholding the image using Otsu's method, interference removal
    light = cv2.threshold(light,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    light = cv2.dilate(light,None,iterations=3)
    light = cv2.erode(light,None,iterations=3)
    light = cv2.dilate(light,None,iterations=1)

    # Calculates horizontal gradient, normalizes values, and converts to uint8 data type.
    gradX = cv2.Sobel(blackHat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX),np.max(gradX))
    gradX = 255*((gradX - minVal)/(maxVal - gradX))
    gradX = gradX.astype("uint8")

    # Blur
    gradX = cv2.GaussianBlur(gradX,(3,3),0)
    gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectangleKernel)
    thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    # Eroding and dilating
    thresh = ErodeDilate(thresh,light)

    # Creating list of candidates
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imu.grab_contours(cnts)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    return (cnts, gray)



listOfPicturesCars = os.listdir('cars')

for nameOfImage in listOfPicturesCars:
    img = cv2.imread('cars//' + nameOfImage)
    (cnts, gray) = SetCandidates(nameOfImage,img)
    PrintingPlate(cnts,gray)



listOfPicturesCars = os.listdir('cars1')

for nameOfImage in listOfPicturesCars:
    img = cv2.imread('cars1//' + nameOfImage)
    (cnts, gray) = SetCandidates(nameOfImage,img)
    PrintingPlate(cnts,gray)









