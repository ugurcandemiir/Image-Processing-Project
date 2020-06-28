import cv2
import numpy as np
import os
import glob
import dlib
# Opening Camera

def Camera(mirror=False):
    cap = cv2.VideoCapture(0)
    return cap
    
    
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def snapshot(frame):

    cv2.imwrite('test.jpg',frame)
    return
        
def main():
    Camera(mirror=True)


if __name__ == '__main__':
    main()

# ===============================

