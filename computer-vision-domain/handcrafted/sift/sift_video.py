import numpy as np
import cv2

filepath = "C:/Users\pc\Downloads\Datasets\cctv\dideoplayback.mp4".replace('\\', '/')

cap = cv2.VideoCapture(filepath)

counter = 0
keypoint_array = []
keypoint_dict = {}

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints = sift.detect(gray, None)
    # print(counter, len(keypoints))
    keypoint_dict[counter] = len(keypoints)

    img = cv2.drawKeypoints(gray, keypoints, frame)

    cv2.imshow('general video', img)

    # if counter == 495:
    #     cv2.imshow('min', img)
    # elif counter == 258:
    #     cv2.imshow('max', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if counter == 600:
        break

    counter += 1

cap.release()
cv2.destroyAllWindows()

maximum = max(keypoint_dict, key=keypoint_dict.get)
print('max', maximum, keypoint_dict[maximum])

minimum = min(keypoint_dict, key=keypoint_dict.get)
print('min', minimum, keypoint_dict[minimum])