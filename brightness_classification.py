"""
Classify whether the image is white or gray at a point

We build a histogram of the brightness changes.
We can assume that the white background which we want to extract
is quite homogenious in change.


Getting the perfect white color of a sheet of paper which is
photographed.

Model:
    brightness = (light_source_brightness / (dx² + dy² + dz²) + ambient_light) * photopaphy_sensor_value
    brightness = (1 / (dx² + dy² + dz²) + ambient_light*light_source_brightness) * (photopaphy_sensor_value/light_source_brightness)
    brightness = (1 / (dx² + dy² + dz²) + b) * c
    brightness = c / (dx² + dy² + dz²) + bc
    brightness = c / (dx² + dy² + dz²) + a

We can remove photopaphy_sensor_value, since this is in the values.
Variables: x, y, z, a, c
Assumptions:
    The image does not reflect. From each angle, we get the same brightness

Image:
    +-----------------+-----------------+
    |                 |                 |
    |       m11       |       m12       |
    |                 |                 |
    +-----------------m-----------------+
    |                 |                 |
    |       m21       |       m22       |
    |                 |                 |
    +-----------------+-----------------+

    where v1 to v4 are generated from the means of their quadrants and
    m is generated from the means of all.

    NOTE: This will not be implemented.
    It gets a bit hard to evaluate.
    As we can assume
    1. a quite big z value:
        beamer is light is further away than the image is big
    2. That y = 1 / (x + z) where x << z can be approximated by a
       linear equation (it flattens out)
    We will not go further with this approach.
    We will gain:
    1. performance as we do not need to solve equations of at least order 2-4
    2. time for implementation as we do not need to solve the equations.
"""
#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.cluster.vq as vq
import time
import sys
from utils import *
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize

def get_foreground_from_BGR(image, x_partitions=5, y_partitions=4):
    """Get the foreground if an BGR image which is a photo lit by a lamp."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return get_foreground_from_gray(gray, x_partitions=x_partitions, y_partitions=y_partitions)

def f(a, c, lx, ly, lz, x, y):
    """Compute the lamp brightness at (x,y).

    See the comment of the module."""
    return c / ((lx-x)**2 + (ly-y)**2 + abs(lz)) + a

def get_foreground_from_gray(gray, x_partitions=5, y_partitions=4):
    """Get the foreground from a gray image of a lamp lit photo."""
    computed_lamp_brightness = compute_lamp_brightness(gray, x_partitions=x_partitions, y_partitions=y_partitions)
    return foreground_from_gray_and_lamp_brightness(gray, computed_lamp_brightness)

def foreground_from_gray_and_lamp_brightness(gray, computed_lamp_brightness):
    """Finally get the foreground from the lamp lit photo without distorting lamp influence."""
    gray_without_lamp = gray - computed_lamp_brightness
    non_zero = cv2.compare(gray, computed_lamp_brightness, cv2.CMP_GE)
    gray_without_lamp = cv2.bitwise_and(non_zero, gray_without_lamp)
    thresh, im_colored = cv2.threshold(gray_without_lamp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_colored

def compute_lamp_brightness(gray, x_partitions=4, y_partitions=4, debug=True):
    """Return the lamp brightness of a gray image.

    See the comment of the module for more information.
    """
    # from https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
    X_PARTITIONS = x_partitions
    Y_PARTITIONS = y_partitions
    DIMENSIONS = (len(gray), len(gray[0]))
    h, w = DIMENSIONS
    inputs = np.array([
        (w * (x + 0.5) / X_PARTITIONS, h * (y + 0.5) / Y_PARTITIONS, xy_split.mean())
        for x, x_split in enumerate(np.array_split(gray, X_PARTITIONS, axis=1))
        for y, xy_split in enumerate(np.array_split(x_split, Y_PARTITIONS, axis=0))
        ])

    def fitness(args):
        a, c, lx, ly, lz = args
        return sum(map(lambda xym: (f(a, c, lx, ly, lz, xym[0], xym[1]) - xym[2])**2, inputs))

    START = (inputs[0][2], 1.0*h*w*h*w, w/2, h/2, w*10)
    result = minimize(fitness, START, method="Nelder-Mead", options={"maxiter":100000, "disp":True})
    a, c, lx, ly, lz = result.x
    NO_LAMP = (a, 0, lx, ly, lz)
    if debug:
        print("inputs", inputs)
        print("result", result, result.x)
        print("fitness", fitness(result.x), "start", fitness(START), "no lamp", fitness(NO_LAMP))
        print("ambient light", a, "lamp at", (lx, ly, abs(lz)**0.5))

    if a < 0:
        # We can not have a negative ambient light intensity.
        return 0

    if lz != lz: # nan
        if debug:
            print("No place for the lamp?")
    return np.array(
        [[f(0, c, lx, ly, lz, x, y) for x in range(w)] for y in range(h)], np.uint8)

if __name__ == "__main__":
    image_path = (sys.argv[1] if len(sys.argv) > 1 else "pathxxx.png")

    t = time.time()
    image = cv2.imread(image_path)
    print("read:", time.time() - t); t = time.time()
    DIMENSIONS = (len(image), len(image[0]))
    print("DIMENSIONS", DIMENSIONS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    computed_lamp_brightness = compute_lamp_brightness(gray)
    print("lamp brightness:", time.time() - t); t = time.time()
    im_colored = foreground_from_gray_and_lamp_brightness(gray, computed_lamp_brightness)
    print("threshold:", time.time() - t); t = time.time()
    image_without_lamp = image - cv2.cvtColor(computed_lamp_brightness, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("im_colored", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("im_colored", im_colored)
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", image)
    cv2.namedWindow("image_without_lamp", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image_without_lamp", image_without_lamp)
    cv2.namedWindow("gray", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("gray", gray)
    print("images:", time.time() - t); t = time.time()

    cv2.waitKey(0)
