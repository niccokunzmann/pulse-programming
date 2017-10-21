#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.cluster.vq as vq
import time
import sys
from utils import *


image_path = (sys.argv[1] if len(sys.argv) > 1 else "pathxxx.png")

codebook_raw = [[0,0,0], [255, 255, 255], [128,128,128]]
NUMBER_OF_COLORS = len(codebook_raw)


image = cv2.imread(image_path)
DIMENSIONS = (len(image), len(image[0]))

#image = cv2.GaussianBlur(image, (5, 5), 0)
flat_colors = image.reshape((-1, 3))
whitened = vq.whiten(np.concatenate((flat_colors, np.array(codebook_raw, np.uint8))))
all_features = whitened[:len(flat_colors)]
codebook_whitened = whitened[len(flat_colors):]
codebook_whitened = len(codebook_whitened)
codebook, distortion = vq.kmeans(whitened, codebook_whitened)
print("codebook", codebook)
code, distance = vq.vq(all_features, codebook)

reshaped_code = code.reshape(DIMENSIONS)
print("Assuming background has biggest area.")

color_usage = [0] * NUMBER_OF_COLORS
for cls in code:
    color_usage[cls] += 1
background_class = max(range(NUMBER_OF_COLORS), key=lambda i: color_usage[i])
print("color_usage", color_usage, "background_class", background_class)
colorful_class = max(range(NUMBER_OF_COLORS), key=lambda i: how_colored_is_this(codebook[i]))
print("colorful_class", colorful_class)
assert colorful_class != background_class, "The background must not be colored."
gate_source_class = colorful_class
road_source_class = (set(range(NUMBER_OF_COLORS)) - set((colorful_class, background_class))).pop()

road_image = np.array(
    [[255 * (col == road_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)
gate_image = np.array(
    [[255 * (col == gate_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)

cv2.namedWindow("road_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("road_image", road_image)
cv2.namedWindow("gate_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("gate_image", gate_image)

# pulse generation
zeros = np.zeros(DIMENSIONS, np.uint8)

road_pulse_element = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
right_pulse_element = np.array([[0,0,0],[1,0,0],[0,0,0]], np.uint8)
down_pulse_element = np.array([[0,1,0],[0,0,0],[0,0,0]], np.uint8)
nor_pulse_element = np.array([[1,0,0],[0,0,0],[0,0,0]], np.uint8)
generator_element1 = np.array([[1,0,0],[0,0,0],[0,0,0]], np.uint8)
generator_element2 = np.array([[0,1,1],[1,1,1],[1,1,1]], np.uint8)
expand_element = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)

# generate image with only generator_element1
pulse_input_image = cv2.subtract(
    cv2.dilate(gate_image, generator_element1),
    cv2.dilate(gate_image, generator_element2))
gate_expanded = cv2.dilate(gate_image, expand_element)

cv2.namedWindow("pulse_input_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("pulse_input_image", pulse_input_image)

def pulse(state, element, area, input=()):
    """Make the next pulse using an element on an area, adding the input
    as a pulse source."""
    pulse1, pulse0 = state
    pulse2 = cv2.dilate(pulse1, element)
    for input_image in input:
        pulse2 = cv2.bitwise_or(pulse2, input_image)
    pulse2 = cv2.subtract(pulse2, pulse0)
    pulse2 = cv2.bitwise_and(pulse2, area)
    return pulse2, pulse1

cv2.namedWindow("W3", cv2.WINDOW_AUTOSIZE)
nor_pulse = down_pulse = right_pulse = road_pulse = (zeros, zeros)
while cv2.waitKey(500) != 27: # press escape
    t = time.time()
    pulse_input = cv2.bitwise_and(pulse_input_image, cv2.bitwise_not(cv2.dilate(nor_pulse[0], expand_element)))
    road_pulse = pulse(road_pulse, road_pulse_element, road_image, (pulse_input, down_pulse[0], right_pulse[0]))
    right_pulse = pulse(right_pulse, right_pulse_element, gate_expanded, (road_pulse[0],))
    down_pulse = pulse(down_pulse, down_pulse_element, gate_expanded, (road_pulse[0],))
    nor_input = cv2.bitwise_and(right_pulse[0], down_pulse[0])
    nor_pulse = pulse(nor_pulse, nor_pulse_element, gate_expanded, (nor_input,))
    image = road_pulse[0]
    for a_pulse in (right_pulse, down_pulse, nor_pulse):
        image = cv2.bitwise_or(image, a_pulse[0])
    cv2.imshow("W3", image)
    print("duration", time.time() - t)
exit(0)
