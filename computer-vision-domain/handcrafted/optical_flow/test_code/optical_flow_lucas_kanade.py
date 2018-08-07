"""
Source:http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

Video tute: https://classroom.udacity.com/courses/ud810/lessons/3260618540/concepts/32740285420923

Lucas-Kanade Optical Flow in OpenCV

Assumptions:
1. The pixel intensities of an object do not change between consecutive frames.
2. Neighbouring pixels have similar motion.

Basically, Lucas-Kanade method computes optical flow for a sparse feature set (in this example, corners detected using
Shi-Tomasi algorithm) Further, this code doesnt check how correct are the next keypoints. So even if any feature point disappears in image,
there is a chance that optical flow finds the next point which may look close to it. So actually for a robust
tracking, corner points should be detected in particular intervals. OpenCV samples comes up with such a sample
which finds the feature points at every 5 frames.
"""

import cv2
import numpy as np
import sys

category = "boxing"
filename = "person04_boxing_d1_uncomp"
# filepath = "../../inputdata/" + category + "/" + filename + ".avi"
# filepath = "C:/Users\pc\Downloads\Datasets\cctv\dideoplayback.mp4".replace('\\', '/')
filepath = "E:\Projects\computer-vision\computer-vision-python\opencv-starter\data/visor_1246523090489_new_6_camera1.avi".replace('\\', '/')

cap = cv2.VideoCapture(filepath)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
