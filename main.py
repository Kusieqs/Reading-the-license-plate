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

listOfPictures = os.listdir('cars')

for nameOfImage in listOfPictures:
    img = cv2.imread('cars//' + nameOfImage)
    ShowImage(nameOfImage,img)
    img = imu.resize(img,width=600)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")

    gray = cv2.filter2D(gray,-1,sharpen)

    rectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19
                                                                 ,13))
    blackHat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,rectangleKernel)
    ShowImage(nameOfImage,blackHat)
    ShowImage(nameOfImage,gray)
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    light = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,squareKernel)
    light = cv2.threshold(light,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    light = cv2.dilate(light,None,iterations=3)
    light = cv2.erode(light,None,iterations=2)

    ShowImage('light',light)

    gradX = cv2.Sobel(blackHat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX),np.max(gradX))
    gradX = 255*((gradX - minVal)/(maxVal - gradX))
    gradX = gradX.astype("uint8")
    ShowImage(nameOfImage,gradX)

    gradX = cv2.GaussianBlur(gradX,(3,3),0)
    gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectangleKernel)
    thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    ShowImage('Przed erode',thresh)

    thresh = cv2.erode(thresh,None,iterations=4)
    thresh = cv2.dilate(thresh,None,iterations=6)
    cv2.imshow(nameOfImage,thresh)
    cv2.imshow('light',light)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresh = cv2.bitwise_and(thresh,thresh,mask=light)
    thresh = cv2.dilate(thresh,None,iterations=4)
    thresh = cv2.erode(thresh,None,iterations=7)
    thresh = cv2.dilate(thresh,None,iterations=7)
    thresh = cv2.erode(thresh,None,iterations=3)
    ShowImage(nameOfImage,thresh)

    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imu.grab_contours(cnts)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

    lpCnt = None
    roi = None

    for c in cnts:

        (x,y,w,h) = cv2.boundingRect(c)
        if w >= 60 and w <= 250:
            lpCnt = c
            plate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            cv2.imshow("Plate", plate)
            cv2.imshow('roi', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            text = pytesseract.image_to_string(roi, config=r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
            # Wyświetlenie tekstu

            if len(text) >=6:
                print(text)



