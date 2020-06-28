import numpy
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
import numpy as np
from camera_opening import *
from mask_gif import *
from detection_expression import *

root=Tk()
root.geometry('500x500')
frame = Frame(root, relief=RIDGE, borderwidth=5)
frame.pack(fill=BOTH,expand=3)
root.title('Flash Photo Booths')
frame.config(background='light blue')
label = Label(frame, text="Flash Photo Booth",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)

def exitt():
   exit()

def open_camera():
   capture =Camera()
   while True:
      ret,frame=capture.read()
      frame = rescale_frame(frame,percent=75)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
      elif cv2.waitKey(32) :
         snapshot(frame)
   capture.release()
   cv2.destroyAllWindows()

def edge_detection():
    capture = Camera()
    while True:
      ret,frame=capture.read()
      frame = rescale_frame(frame,percent=75)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = cv2.Canny(gray,100,300)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
      elif cv2.waitKey(32) :
         snapshot(frame)

    capture.release()
    cv2.destroyAllWindows()

def emboss():
    capture = Camera()
    kernel = np.array([[0,-1,-1],
    [1,0,-1],
    [1,1,0]])
    while True:
       ret,frame=capture.read()
       frame = rescale_frame(frame,percent=75)
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       frame = cv2.filter2D(gray, -1, kernel)
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xFF ==ord('q'):
          break
       elif cv2.waitKey(32) :
          snapshot(frame)
    capture.release()
    cv2.destroyAllWindows()

def sepia():
    capture =Camera()
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    while True:
       ret,frame=capture.read()
       frame = rescale_frame(frame,percent=75)
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       frame = cv2.filter2D(gray, -1, kernel)
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xFF ==ord('q'):
          break
       elif cv2.waitKey(32) :
          snapshot(frame)
    capture.release()
    cv2.destroyAllWindows()

def vintage_filter():
    cap = Camera()
    while True:

        ret, frame = cap.read()
        frame = rescale_frame(frame,percent=75)
        # Filtering Part
        rows, cols = frame.shape[:2]
        # Create a Gaussian filter
        kernel_x = cv2.getGaussianKernel(cols,200)
        kernel_y = cv2.getGaussianKernel(rows,200)
        kernel = kernel_y * kernel_x.T
        filter = 255 * kernel / np.linalg.norm(kernel)
        vintage_im = np.copy(frame)
        # for each channel in the input image, we will apply the above filter
        for i in range(3):
            vintage_im[:,:,i] = vintage_im[:,:,i] * filter

        cv2.imshow('frame',vintage_im)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        elif cv2.waitKey(32) :
            snapshot(frame)
    cap.release()
    cv2.destroyAllWindows()

def gaussian():
    capture =Camera()
    while True:
       ret,frame=capture.read()
       frame = rescale_frame(frame,percent=75)
       vintage_im = cv2.GaussianBlur(frame,(5,5),cv2.COLOR_BGR2GRAY)
       cv2.imshow('frame',vintage_im)
       if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
       elif cv2.waitKey(32) :
            snapshot(frame)
    capture.release()
    cv2.destroyAllWindows()

def corona_mask():
    cap = Camera()
    while True:
       frame = adding_mask(cap)
       frame = rescale_frame(frame,percent=75)
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
       elif cv2.waitKey(32) :
            snapshot(frame)
    cap.release()
    cv2.destroyAllWindows()

def apply_hue_saturation():
    cap = Camera()
    while True:
        ret,frame=cap.read()

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        s.fill(199)
        v.fill(255)
        hsv_image = cv2.merge([h, s, v])
        out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.addWeighted(out, 0.25, frame, 1.0, .23, frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        elif cv2.waitKey(32) :
            snapshot(frame)
    cap.release()
    cv2.destroyAllWindows()

def alpha_blend(frame_1, frame_2, mask):
    alpha = mask/255.0
    blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
    return blended

def portrait_mode():

    cap = Camera()
    while True:
        ret,frame=cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
        blured = cv2.GaussianBlur(frame, (21,21), 11)
        blended = alpha_blend(frame, blured, mask)
        frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        elif cv2.waitKey(32) :
            snapshot(frame)
    cap.release()
    cv2.destroyAllWindows()

def expression():
    cap = Camera()
    while True:
       frame = expression_detection(cap)
       frame = rescale_frame(frame,percent=75)
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
       elif cv2.waitKey(32) :
            snapshot(frame)

    cap.release()
    cv2.destroyAllWindows()



but1=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=open_camera,text='Open Cam',font=('helvetica 15 bold'))
but1.place(x=40,y=100)

but2=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=corona_mask,text='Corona Vibes',font=('helvetica 15 bold'))
but2.place(x=260,y=100)

but3=Button(frame,padx=5,pady=5,width=20,bg='white',fg='black',relief=GROOVE,command=expression,text='Face Expression \n and Detection' ,font=('helvetica 15 bold'))
but3.place(x=40,y=160)

but4=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=sepia,text='Sepia Filter ',font=('helvetica 15 bold'))
but4.place(x=260,y=160)

but5=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=vintage_filter,text='Vintage Filter',font=('helvetica 15 bold'))
but5.place(x=40,y=220)

but6=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=gaussian,text='Gaussian Blurring Filter',font=('helvetica 15 bold'))
but6.place(x=260,y=220)

but7=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=edge_detection,text='Edge Detection',font=('helvetica 15 bold'))
but7.place(x=40,y=280)

but8=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=emboss,text='Emboss Filter',font=('helvetica 15 bold'))
but8.place(x=260,y=280)

but9=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=apply_hue_saturation ,text='Saturation',font=('helvetica 15 bold'))
but9.place(x=40,y=340)

but10=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,command=portrait_mode ,text='Portrait',font=('helvetica 15 bold'))
but10.place(x=260,y=340)


but11=Button(frame,padx=5,pady=5,height = 2,width=20,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but11.place(x=150,y=400)


root.mainloop()
