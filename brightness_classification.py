"""
Classify whether the image is white or gray at a point

We build a histogram of the brightness changes.
We can assume that the white background which we want to extract
is quite homogenious in change.

"""
#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.cluster.vq as vq
import time
import sys
from utils import *
from sklearn.mixture import GaussianMixture

image_path = (sys.argv[1] if len(sys.argv) > 1 else "pathxxx.png")

t = time.time()
image = cv2.imread(image_path)
print("read:", time.time() - t); t = time.time()
DIMENSIONS = (len(image), len(image[0]))

b,g,r = cv2.split(np.array(image, np.int16))
change_kernel1 = np.array([[1,3,0], [3,0,0], [0,0,0]], np.uint8)
change_kernel2 = np.array([[0,0,0], [0,0,3], [1,3,0]], np.uint8)
b_change, g_change, r_change = map(lambda color:
    cv2.absdiff(
        cv2.filter2D(color, -1, change_kernel1),
        cv2.filter2D(color, -1, change_kernel2)
    ), (b, g, r))
print("coloring:", time.time() - t); t = time.time()
color_change = r_change + b_change + g_change

cv2.namedWindow("color_change", cv2.WINDOW_AUTOSIZE)
cv2.imshow("color_change", color_change * 50)
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("image", image)
print("images:", time.time() - t); t = time.time()
print(set(color_change.reshape(-1)))
cv2.waitKey(0)
