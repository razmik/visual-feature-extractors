"""
Distinctive Image Features from Scale-Invariant Keypoints, which extract keypoints and compute its descriptors.

There are mainly four steps involved in SIFT algorithm. We will see them one-by-one.

Use to recognize a new image based on a exisiting image. This match features invariant to scaling.

Description: https://www.youtube.com/watch?v=U0wqePj4Mx0
Tute: http://docs.opencv.org/trunk/da/df5/tutorial_py_sift_intro.html
Steps:
1. Create a scale space of images.
        Construct a set of progressively gaussian blurred images
        Take differences to get a difference of gaussian pyramid
2. Find local extrema in this scale-space. Choose keypoints from the extrema. (min and max extremas)
3. Keypoint localization.
        Taylor series expansion of scale space to get more accurate location of extrema,
        and if the intensity at this extrema is less than a threshold value (0.03 as per the paper), it is rejected.
        DoG has higher response for edges, so edges also need to be removed. For this, a concept similar to Harris
        corner detector is used. They used a 2x2 Hessian matrix (H) to compute the principal curvature.
4. Orientation Assignment
        orientation is assigned to each keypoint to achieve invariance to image rotation. A neigbourhood is taken around
        the keypoint location depending on the scale, and the gradient magnitude and direction is calculated in that
        region. An orientation histogram with 36 bins covering 360 degrees is created. The highest peak in the histogram
        is taken and any peak above 80% of it is also considered to calculate the orientation.
5. For each keypoint, in a 16x16 window, find histograms of gradient directions.
6. Create a feature vector out of these
7 Keypoint matching
        Keypoints between two images are matched by identifying their nearest neighbours. But in some cases,
        the second closest-match may be very near to the first. It may happen due to noise or some other reasons.
        In that case, ratio of closest-distance to second-closest distance is taken. If it is greater than 0.8,
        they are rejected. It eliminaters around 90% of false matches while discards only 5% correct matches,
        as per the paper.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

filepath = "images\cctv_mid_res.jpg".replace('\\', '/')  # 2128 keypoints
# filepath = "images\maxresdefault.jpg".replace('\\', '/')  # 27126 keypoints

img = cv2.imread(filepath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# keypoints = sift.detect(gray, None)
# img = cv2.drawKeypoints(gray, keypoints, img)

blur = cv2.GaussianBlur(gray,(5,5),0)

blur2 = cv2.GaussianBlur(blur,(5,5),0)

diff = gray - blur
diff2 = blur - blur2

plt.figure(1)
plt.imshow(gray, cmap='gray')

plt.figure(2)
plt.imshow(blur, cmap='gray')

plt.figure(3)
plt.imshow(diff, cmap='gray')

plt.figure(4)
plt.imshow((diff+blur), cmap='gray')

plt.show()

# cv2.imwrite('images\sift_keypoints.jpg', img)
