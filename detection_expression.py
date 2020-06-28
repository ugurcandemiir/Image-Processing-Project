
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
import glob
import dlib
from math import hypot
import face_recognition



face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']


#make array of sample pictures with encodings
known_face_encodings = []
known_face_names = []

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'known_people/')
#make an array of all the saved jpg files' paths
list_of_files = [f for f in glob.glob(path+'*.jpg')]
#find number of known faces
number_files = len(list_of_files)

names = list_of_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i]) #Loads an image file (.jpg, .png, etc) into a numpy array
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0] #Given an image, return the 128-dimension face encoding for each face in the image
    known_face_encodings.append(globals()['image_encoding_{}'.format(i)])

    # Create array of known names
    names[i] = names[i].replace("known_people/", "")
    known_face_names.append(names[i])

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def expression_detection(cap):

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # ==========  NEW PART  ============
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # ==========  NEW PART  ============

        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
        # rect,face,image = face_detector(frame)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)

            if (label == "Happy"):
                face_locations = face_recognition.face_locations(rgb_small_frame) #Returns an array of bounding
                # boxes of human faces in a image
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding) #Compare
                    # a list of face encodings against a candidate encoding to see if they match.
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    #Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
                    # for each comparison face. The distance tells you how similar the faces are
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                    tmp = name.split("/")
                    name = tmp[-1]
                    label = label + " " + name
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)


            else:
                    text = "I cannot detect your face unless you SMILE !!!"
                    cv2.putText(frame, text,label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)



        return frame
