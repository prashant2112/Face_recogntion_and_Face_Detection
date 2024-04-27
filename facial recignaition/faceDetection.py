import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0) # captures video
cap.set(3,640)# set Width to 640
cap.set(4,480)# set Height to 480

while True: #stays on until we break it
    ret, img = cap.read() # Reading every signal frame from the video capture
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts the resize colour image to grayscale
    
    faces = faceCascade.detectMultiScale( # parameters that will vary depending on the size of the image
            gray, scaleFactor=1.01, minNeighbors=10, minSize=(30,30)) 

    for (x,y,w,h) in faces: #face detection rectanle/box around the face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('Video',img)
    B = cv2.waitKey(2) & 0xff
    if B == 27: #Press 'ESC' to quit
        break
    
cap.release()
cv2.destroyAllWindows()
    
        
    




















