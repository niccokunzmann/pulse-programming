#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.cluster.vq as vq

NUMBER_OF_COLORS = 2
NUMBER_OF_COLORS += 1 # add background color

image = cv2.imread("pathxxx.png")
DIMENSIONS = (len(image), len(image[0]))

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
reshaped_code = code.reshape(DIMENSIONS)
print("Assuming background has biggest area.")
counts = [0] * NUMBER_OF_COLORS
for cls in code:
    counts[cls] += 1
color_usage_indices = list(range(NUMBER_OF_COLORS))
color_usage_indices.sort(key=lambda i:counts[i])
background_class = color_usage_indices[-1]
road_class = color_usage_indices[-2]
pulse_source_class = color_usage_indices[-3]
road_image = np.array(
    [[255 * (col == road_class) for col in row]
     for row in reshaped_code],
    np.uint8)
pulse_source_image = np.array(
    [[255 * (col == pulse_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)

cv2.namedWindow("road_image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("pulse_source_image", cv2.WINDOW_AUTOSIZE)

cv2.imshow("road_image", road_image)
cv2.imshow("pulse_source_image", pulse_source_image)

# pulse generation
pulse0 = zeros = np.zeros(DIMENSIONS, np.uint8)
pulse1 = np.zeros(DIMENSIONS, np.uint8)

element = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

def pulse(state):
    pulse1, pulse0 = state
    pulse2 = cv2.dilate(pulse1, element)
    pulse2 = cv2.subtract(pulse2, pulse1)
    pulse2 = cv2.subtract(pulse2, pulse0)
    pulse2 = cv2.multiply(pulse2, 1)
    pulse2 = cv2.bitwise_and(pulse2, expansion_room)
    return pulse2, pulse1

road_image_negative = cv2.bitwise_not(road_image)
border = cv2.dilate(road_image, element)
border = cv2.subtract(border, img)
cv2.namedWindow("border", cv2.WINDOW_AUTOSIZE)
cv2.imshow("border", border)
expansion_room_128 = cv2.bitwise_and(road_image, np.full(DIMENSIONS, 128, np.uint8))
#expansion_room = cv2.bitwise_or(expansion_room_128, expansion_room_64)
expansion_room = expansion_room_128
cv2.namedWindow("expansion_room", cv2.WINDOW_AUTOSIZE)
cv2.imshow("expansion_room", expansion_room)
state0 = state = pulse1, pulse0
cv2.namedWindow("W3", cv2.WINDOW_AUTOSIZE)
while cv2.waitKey(200) != 27: # press escape
    cv2.imshow("W3", cv2.multiply(state[0], 4))
    state = pulse(state)
    print(list(map(bin, set(list(state[0].reshape((-1)))))))
    if cv2.countNonZero(state[0]) == 0:
        state = state0
