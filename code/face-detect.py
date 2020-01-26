'''
@author : aalfianrachmat@gmail.com
@date   : 2020-01-26

This code is used to acquire faces dataset using HaarCascade and
openCV library.
'''

import cv2
import numpy as np
import os, shutil

# List of available faces dataset
labels = ['person1',
          'person2']

# Currently used label
label = labels[0] # <<<--------- Change label here

path = "../dataset/{}".format(label)
if not (os.path.exists(path = "../dataset/{}".format(label))):
    os.mkdir(path)

for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))




faceCascade = cv2.CascadeClassifier('../detector_architectures/haarcascade_frontalface_default.xml')

# Set video source to the default webcam
video_capture = cv2.VideoCapture(0)

img_counter = 0; # Initialize counter: img_counter
# Capture the video
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert webcam feed to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
            gray
            )
    # Create rectangle where faces are found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-100,y-100), (x+w+100, y+h+100), (0, 255, 0), 2)
        
    # Show image
    cv2.imshow('FaceDetection', frame)
     
    # Saved image
    # SPACE pressed
    if (len(faces) > 0 and cv2.waitKey(25)%256 == 32):
        img_size  = gray.shape;
        gray = gray[max(0,y-100) : min(img_size[0], y+h+100), 
                    max(0,x-100) : min(img_size[1], x+w+100)]

        img_name = "/facedetect_webcam_{}.png".format(img_counter)
        img_save = cv2.resize(gray, (128,128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path + img_name, gray)
        print("{} written".format(img_name))
        img_counter +=1

    # Escape method
    # ESC Pressed
    elif (cv2.waitKey(25)%256 == 27):
        print("Closing.....")
        break
        
# Clean everything and release the capture
video_capture.release()
cv2.destroyAllWindows()