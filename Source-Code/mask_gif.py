import cv2
import numpy as np
import cv2
import numpy as np
import os
import glob
import dlib
from math import hypot
import face_recognition



# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mask_image = cv2.imread("medical_mask.png")


def adding_mask(cap):
    ret, frame = cap.read()
    rows, cols, ret = frame.shape
    face_mask = np.zeros((rows, cols), np.uint8)
    while True:
        ret, frame = cap.read()
        face_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Nose coordinates
            top_nose = (landmarks.part(28).x, landmarks.part(28).y)
            center_nose = (landmarks.part(66).x, landmarks.part(66).y)
            left_chick = (landmarks.part(3).x, landmarks.part(3).y)
            right_chick = (landmarks.part(13).x, landmarks.part(13).y)

            nose_width = int(hypot(left_chick[0] - right_chick[0],
                               left_chick[1] - right_chick[1]) * 1.2)
            nose_height = int(nose_width * 0.65)

            # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                                  int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                           int(center_nose[1] + nose_height / 2))


            # Adding the new nose
            nose_mask = cv2.resize(mask_image, (nose_width, nose_height))
            nose_mask_gray = cv2.cvtColor(nose_mask, cv2.COLOR_BGR2GRAY)
            ret, face_mask = cv2.threshold(nose_mask_gray, 25, 255, cv2.THRESH_BINARY_INV)

            mask_area = frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width]
            mask_area_no_nose = cv2.bitwise_and(mask_area, mask_area, mask=face_mask)
            final_nose = cv2.add(mask_area_no_nose, nose_mask)

            frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width] = final_nose
            cv2.putText(frame, "LET'S BEAT CORONAVIRUS!!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,153,0),2)

                        
            return frame

