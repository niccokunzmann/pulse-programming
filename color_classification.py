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


def get_blue_recognition(image, threshold=0.97):
    """Return the places which are blue as binary array.

    Parameters:
    - image
        is the image in BGR format.
        Recommendations:
            If you blur the image before you input it, it yields better results.
            The image does not need to be brighness adjusted.

    Result:
    - is_blue
        An uint8 numpy.array with the dimensions of the image.
    """
    b,g,r = cv2.split(image)
    r = r / 2
    g = g / 2
    a = r + g
    b = b * threshold
    return cv2.compare(b, a, cv2.CMP_GT)

def get_green_recognition(image, threshold=0.97):
    """Return the places which are green as binary array.

    Parameters:
    - image
        is the image in BGR format.
        Recommendations:
            If you blur the image before you input it, it yields better results.
            The image does not need to be brighness adjusted.

    Result:
    - is_green
        An uint8 numpy.array with the dimensions of the image.
    """
    b,g,r = cv2.split(image)
    r = r / 4
    b = b / 4
    a = r + b
    g = np.array(g * (float(threshold) / 2) + 128 - a, np.uint8)
    thresh, is_green = cv2.threshold(g, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return is_green
    #return cv2.compare(g, a, cv2.CMP_GT)


if __name__ == "__main__":
    from camera_image import camera_or_argument_image

    t = time.time()
    image = camera_or_argument_image()
    print("read:", time.time() - t); t = time.time()
    print("blur:", time.time() - t); t = time.time()
    DIMENSIONS = (len(image), len(image[0]))

    blue_img = get_blue_recognition(image, 0.57)
    print("blueness:", time.time() - t); t = time.time()


    cv2.namedWindow("blue_img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("blue_img", blue_img)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", image)
    print("images:", time.time() - t); t = time.time()

    cv2.waitKey(0)
