import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread('C:/Users/91620/Desktop/tooooooooo/Face-Recognition-main/Program/Face/testimages/Kangana.jpg')#test_img path
faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

#Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces,faceID = fr.ID_for_training_database('C:/Users/91620/Desktop/tooooooooo/Face-Recognition-main/Program/Database_trainingimages')
face_recognizer = fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')

name={0:"Talha",1:"Priyanka", 2: "Kangana"} #creating dictionary containing names for each ID's


for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    ID,confidence = face_recognizer.predict(roi_gray)#predicting the label of given image
    print("Confidence: ",confidence)
    print("ID: ",ID)
    fr.draw_rect(test_img,face)
    predicted_name = name[ID]
    if(confidence>42): #If confidence more than 42 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(640,480))
cv2.imshow("face Recognition",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
