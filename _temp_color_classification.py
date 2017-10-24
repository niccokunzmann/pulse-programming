"""
Classify a color and a gray tone.

The color can be classifiedd by looking at the distribution of the different
values: one will spike out.

On the color histogram, we can use the same as the gray-ness-detection.
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
image = cv2.GaussianBlur(image, (5, 5), 0)
print("blur:", time.time() - t); t = time.time()
DIMENSIONS = (len(image), len(image[0]))

def squared(a):
    return cv2.multiply(a, a)
b,g,r = cv2.split(np.array(image / 2, np.int16))
colored_ness = squared(b - g) + squared(r - g) + squared(r - b)
colored_ness2 = colored_ness.reshape(-1, 1)
#colored_ness = np.array([[how_colored_is_this_normalized(c) for c in row] for row in image])
gmm = GaussianMixture(n_components=2).fit(colored_ness2)
gate_image = np.array(gmm.predict(colored_ness2).reshape(DIMENSIONS), np.uint8)
gate_image *= 255
print("coloring:", time.time() - t); t = time.time()
print("gate_image", gate_image)

cv2.namedWindow("colored_ness", cv2.WINDOW_AUTOSIZE)
cv2.imshow("gate_image", gate_image)
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("image", image)
cv2.namedWindow("colored_ness", cv2.WINDOW_AUTOSIZE)
cv2.imshow("colored_ness", colored_ness * 20)
print("images:", time.time() - t); t = time.time()

cv2.waitKey(0)
