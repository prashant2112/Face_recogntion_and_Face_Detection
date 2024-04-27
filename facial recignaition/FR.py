import numpy as np
import sys
import cv2

#def faceDetection(test_img):
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('E:\Collage\2019-2020\Semestor 2\Final Year Project\facial recignaition\tt\Photo2.jpg', 1) 
resize_img = cv2.resize(img, (1100, 720))
cv2.imshow('Photo3', resize_img)

gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray,scaleFactor=1.35,minNeighbors=5,minSize=(30,30))
#return faces, gray_img

for(x,y,w,h) in faces:
    windowName=None
    cv2.rectangle(resize_img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Face Detected Photo",resize_img)
    cv2.waitKey(0)
   #return faces,gray_img
    
cv2.destroyAllWindows()
    
        
    




















