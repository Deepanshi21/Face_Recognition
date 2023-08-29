import cv2
import numpy as np
import face_recognition
import os

path = 'data'
#folder name
images = []
names = []
#separate lists for images and names
LList = os.listdir(path)
print(LList)
#LList contains names of the files

for nm in LList: #iteration
    curImg = cv2.imread(f'{path}/{nm}')
    images.append(curImg)
    names.append(os.path.splitext(nm)[0]) #getting only filename
print(names)
#only names

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] 
        encodeList.append(encode)
    return encodeList
##Converting the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

encodeListKnown = findEncodings(images) #finding encoding for each image
#128 feature points
print("encoding complete")

cap = cv2.VideoCapture(0)
#opening camera

while True:
    success, img = cap.read()
    #success is for True
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #resizing frame for a faster speed
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #again converting

    facesCurFrame = face_recognition.face_locations(imgS) #current video frame
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) #encoded image frame
    #128 dimension face 
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame): #matching in both the list simultaneously
        #matches = face_recognition.compare_faces(encodeListKnown, encodeFace) 
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) 
        #use the known face with the smallest distance to the new face
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex]<0.60:
            name= names[matchIndex].upper()

        else:
            name ='Unknown'

        #if matches[matchIndex]:
            #name = names[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 #resizing
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)



