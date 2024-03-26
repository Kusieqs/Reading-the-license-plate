import os
import cv2
import imutils as imu
import numpy as np
import pytesseract
import skimage.segmentation

def ShowImage(name,img):
    cv2.imshow(f'{name}',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def PrintingPlate(cnts, gray):
    for k in cnts:
        (x,y,w,h) = cv2.boundingRect(k)

        if w >= 60 and w <= 350:

            plate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            roi = imu.resize(roi, width=350)
            roi = cv2.erode(roi, None, iterations= 2)
            roi = cv2.dilate(roi, None, iterations = 3)
            pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            text = pytesseract.image_to_string(roi, config=r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')

            if len(text) >=6:
                print(text)
                cv2.imshow("Plate", plate)
                cv2.imshow('roi', roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
def ErodeDilate(thresh,light):
    thresh = cv2.erode(thresh,None,iterations=3)
    thresh = cv2.dilate(thresh,None,iterations=5)
    ShowImage("1",thresh)
    ShowImage("light",light)
    thresh = cv2.bitwise_and(thresh,thresh,mask=light)
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.erode(thresh,None,iterations=7)
    thresh = cv2.dilate(thresh, None, iterations=7)
    thresh = cv2.erode(thresh,None,iterations=3)
    ShowImage("2",thresh)
    return thresh

def SetCandidates(nameOfImage, img, numb):

    ShowImage(nameOfImage,img)

    # Resizes the image to a width of 600px, Convert the resized image to grayscale, Applies a sharpening filter to the grayscale
    img = imu.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    sharp = cv2.filter2D(gray, -1 , sharpen)

    # Morphological blackhat operation
    rectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,13))
    blackHat = cv2.morphologyEx(sharp,cv2.MORPH_BLACKHAT,rectangleKernel)

    # Closing morphology operation.
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    light = cv2.morphologyEx(sharp,cv2.MORPH_CLOSE,squareKernel)

    # Thresholding the image using Otsu's method, interference removal
    light = cv2.threshold(light,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    light = cv2.dilate(light,None,iterations=3)
    light = cv2.erode(light,None,iterations=3)
    ShowImage('Thresholding after erode and dilate',light)

    # Calculates horizontal gradient, normalizes values, and converts to uint8 data type.
    gradX = cv2.Sobel(blackHat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX),np.max(gradX))
    gradX = 255*((gradX - minVal)/(maxVal - gradX))
    gradX = gradX.astype("uint8")
    ShowImage("gradX",gradX)

    # Blur
    gradX = cv2.GaussianBlur(gradX,(3,3),0)
    gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectangleKernel)
    thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    # Eroding and dilating
    thresh = ErodeDilate(thresh,light)
    ShowImage(nameOfImage,thresh)

    # Creating list of candidates
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imu.grab_contours(cnts)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    return (cnts, gray)



listOfPictures = os.listdir('cars')
choose = 0


for nameOfImage in listOfPictures:
    img = cv2.imread('cars//' + nameOfImage)
    (cnts, gray) = SetCandidates(nameOfImage,img,choose)
    PrintingPlate(cnts,gray)








