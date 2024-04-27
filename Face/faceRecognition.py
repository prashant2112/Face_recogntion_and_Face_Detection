import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('C:/Users/91620/Desktop/tooooooooo/Face-Recognition-main/Program/Face/haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=10,minSize=(30,30),flags=0)

    return faces,gray_img

def ID_for_training_database(directory): #passing tranning dataase directory
    faces=[]#face names
    faceID=[]#face id's

    for path,subdirnames,filenames in os.walk(directory):#it fit a directory and it give us
        #a path a sub directory and file name and to handle  this  we used a os.walk
        
        for filename in filenames: #
            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith .
                continue
           
            id=os.path.basename(path)#it extrects ids so we can use path
            img_path = os.path.join(path, filename) #image path which is required to feed in to the classifer
            print("img_path:",img_path)#printing for debbuging purposess
            print("id:",id)
            test_img = cv2.imread(img_path)#taakes the image path
            if test_img is None:#if the image is not loaded properly it might halt the process
                print("Image not loaded properly")
                continue
            
             #Calling faceDetection function to return faces detected in particular image
            faces_rect,gray_img = faceDetection(test_img)
            if len(faces_rect)!=1:#should be one id's images in the folder otherwise it will confuse the classifer
               continue #Since we are assuming only single person images are being fed to classifier
            (x,y,w,h)=faces_rect[0] # retirned by our faces = 0 entry return a list
            roi_gray=gray_img[y:y+w,x:x+h] #croppes region of interest that is face area from grayscale image and only feddes a face to our classifer
            faces.append(roi_gray)# adds a single item to the existing list
            faceID.append(int(id)) # classider will only take id type int, it converts to id
    return faces,faceID

#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    #it create in to 3x3 image pixel and it write down all the pixel value then it compares the central pixel with sourounding pixels
    #if the surrounding values are lesser then the cetral pixel it drews zero to that box
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=2)


def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),thickness=2)
    

    