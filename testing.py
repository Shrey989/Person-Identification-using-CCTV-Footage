# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:15:45 2020

@author: Abhijeet
"""

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import functools 

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('facefeatures_new_model_3.h5')#facefeatures_new_model2 face_62_500  face2_new_l

class_labels = ['hr','Adish Sthalekar','aditya chaudhary','aditya mukund','Akash','anushka','arnold','ashish','shreyas',
         'Bhushan Kolhe','brayan','chaitali','dane','utkarsh','divyashree','dube','elton','goofran','nilesh','abhijeet',
         'abhijeet','harshala','kevin','madhura','manisha','maria','mihir','mrunal','NeilJason',
         'nikita','harsh','osama','pooja','Poonam Mam','pransu','prathmesh','uttam','RadaliaDSouza',
         'sumantu','ruban','bhoomi','Rushikesh','ryan','salman','Samuel Pais','Satish Sir','sayali','sharayu',
         'divya','umesh','Shreyas Kulkarni',
         'shubham','siddhant','siddharth pagare','rohit','umesh','utkarsh','Pratik shetty','vaishali','vaishnavi','vedant','vivek']



cap = cv2.VideoCapture('abhi.mp4')


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = frame[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(224,224),interpolation=cv2.INTER_LINEAR)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray.all])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            #print(roi)
            roi = np.expand_dims(roi,axis=0)
            #print(roi)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)
            a = np.argmax(preds)
            
            label=class_labels[a]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
