import numpy as np
import cv2

faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('Photos\Photo4.jpg', 1) 
resize_img = cv2.resize(img, (640, 480)) # resizing this image to 640x480
cv2.imshow('photo', resize_img) # it shows the resize image

gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY) # converts the resize colour image to grayscale

faces = faceCascade.detectMultiScale(# parameters that will vary depending on the size of the image
        gray,scaleFactor=1.32,minNeighbors=5,minSize=(10,10))

for(x,y,w,h) in faces: #face detection rectanle/box around the face
    cv2.rectangle(resize_img,(x,y),(x+w,y+h),(255,0,0),2) 
    cv2.imshow("Face Detected Photo",resize_img)
    if cv2.waitKey(0) == ord('q'):#wait until 'q' key is pressed
        break
    
cv2.destroyAllWindows()
    
        
    




















