# import libraries of python opencv and other useful libraries
import cv2
import numpy as np
import time
#from picamera import PiCamera
from datetime import datetime, timedelta


'''This code processes a 30fps video in such a way to process only every 30th frame of it. 
This code assumes that every second the cars are being replaced, and so it does not track them, but just counts
the number of objects recognized as cars every second. Then it sums up all those numbers in the Counter variable'''

'''
camera = PiCamera()
camera.framerate = 1 # use 1 frame per second
#camera.resolution = (320, 240) # optionally decrease the camera resolution
camera.start_preview()
camera.start_recording('/home/pi/Desktop/CarDetection/recording.h264') #save the recording to a file
time.sleep(20) # record the video for 20 seconds only
camera.stop_recording()
camera.stop_preview()
'''

# create VideoCapture object and read from the video file created above
cap = cv2.VideoCapture('cars.mp4')
# use XML classifier already trained on cars
car_cascade = cv2.CascadeClassifier('cars.xml')

Counter = 0 # set the counter for the cars

framenum = 0
while cap.isOpened(): #read until 20 frames
    framenum += 1 #increment the frame counter
    Counter = 0
    #capture frame by frame
    ret, frame = cap.read()

    if ret and framenum == 30:
        framenum = 0
        # convert video into gray scale of each frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect cars in the video
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        # to draw arectangle in each cars and draw a dot at the centroid
        for (x,y,w,h) in cars:
            Counter += 1 # increment the counter for each counted car
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            # find object's centroid
            CoordXCentroid = (x + x + w) // 2
            CoordYCentroid = (y + y + h) // 2
            ObjectCentroid = (CoordXCentroid, CoordYCentroid)
            cv2.circle(frame, ObjectCentroid, 1, (0, 0, 0), 5)

        # display the resulting frame
        cv2.imshow('video', frame)

        Res = open("results.txt", "a")  # open the output file for appending

        # write the result with the time of the recording to the output file:
        Res.write("Results for " + datetime.now().strftime("%I:%M%p on %B %d, %Y") + " Cars in frame = " + str(Counter) + "\n")
    # press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
# release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()

