#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.cluster.vq as vq

NUMBER_OF_COLORS = 20
NUMBER_OF_COLORS += 1 # add background color

image = cv2.imread("example.png")

image = cv2.GaussianBlur(image, (5, 5), 0)

flat_colors = image.reshape((-1, 3))
maximum_bright_colors = flat_colors
#def maximize_color(color):
#    m = color.sum() / 3
#    if m == 0:
#        return (255, 255, 255)
#    return [int(v * 255 / 3 // m) for v in color]
#maximum_bright_colors = np.array(list(map(maximize_color, flat_colors)), flat_colors.dtype)
print("flat_colors", set(map(tuple, flat_colors)))
whitened_colors = vq.whiten(maximum_bright_colors)
codebook, distortion = vq.kmeans(whitened_colors, NUMBER_OF_COLORS)
print("codebook", codebook, "distortion", distortion)
code, distance = vq.vq(whitened_colors, codebook)
reshaped_code = code.reshape((len(image), len(image[0])))
print("Assuming background has biggest area.")
counts = [0] * NUMBER_OF_COLORS
for cls in code:
    counts[cls] += 1
max_class = max(range(NUMBER_OF_COLORS), key=lambda i:counts[i])
classified_image = np.array(
    [[255 * (col != max_class) for col in row]
     for row in reshaped_code],
    np.uint8)

print("classes", set(classified_image.flatten()))
print("classified_image", classified_image)

img = classified_image

element = np.array([[0,0,0],[0,0,1],[1,1,1]], np.uint8)

# compute tops
img2 = cv2.dilate(img, element)
img2 = cv2.subtract(img, img2)
element2 = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
img2 = cv2.dilate(img2, element2, iterations=20)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=75,param2=25,minRadius=20,maxRadius=200)
print("circles", circles)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img2,(i[0],i[1]),i[2],128,2)
    # draw the center of the circle
    cv2.circle(img2,(i[0],i[1]),2,128,3)

cv2.namedWindow("W1", cv2.WINDOW_AUTOSIZE)
cv2.imshow("W1", img)

cv2.namedWindow("W2", cv2.WINDOW_AUTOSIZE)
cv2.imshow("W2", img2)

cv2.waitKey(0)
