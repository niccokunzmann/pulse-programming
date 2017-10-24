#!/usr/bin/env python3
import os
import numpy as np
import cv2
import scipy.cluster.vq as vq
import time
import sys
from utils import *
from color_classification import get_blue_recognition
from brightness_classification import get_foreground_from_BGR

image_path = (sys.argv[1] if len(sys.argv) > 1 else "explanation.png")

codebook_raw = [[0,0,0], [255, 255, 255], [128,128,128]]
NUMBER_OF_COLORS = len(codebook_raw)
OUTPUT_ANIMATION = False

image = camera_image = cv2.imread(image_path)
DIMENSIONS = (len(image), len(image[0]))

#image = cv2.GaussianBlur(image, (5, 5), 0)
gate_image = get_blue_recognition(image)
foreground_image = get_foreground_from_BGR(image)
road_image = cv2.bitwise_and(cv2.bitwise_not(foreground_image), cv2.bitwise_not(gate_image))

DELATION_ITERATIONS = 3
EROSION_ITERATIONS = 2
expansion_element = np.array([[1,1,1], [1,1,1], [1,1,1]])
gate_image = cv2.erode(gate_image, expansion_element, iterations=EROSION_ITERATIONS)
gate_image = cv2.dilate(gate_image, expansion_element, iterations=DELATION_ITERATIONS)
road_image = cv2.dilate(road_image, expansion_element, iterations=DELATION_ITERATIONS)
road_image = cv2.erode(road_image, expansion_element, iterations=EROSION_ITERATIONS)

road_image = cv2.bitwise_and(road_image, cv2.bitwise_not(gate_image))


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
pulse_input_element = np.array([[0,0,1],[0,0,1],[1,1,1]], np.uint8)
expand_element = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
gate_expand_element = np.array([[1,1,1],[1,1,0],[1,0,0]], np.uint8)

# generate image with only generator_element1
pulse_input_image = cv2.subtract(
    gate_image,
    cv2.dilate(gate_image, pulse_input_element))
gate_expanded = cv2.dilate(gate_image, gate_expand_element)

#cv2.namedWindow("pulse_input_image", cv2.WINDOW_AUTOSIZE)
#cv2.imshow("pulse_input_image", pulse_input_image)

def output_animation(frame_index, base_path, image):
    filename, extension = os.path.splitext(base_path)
    filename += "-animation-frame-{:05}{}".format(frame_index, extension)
    cv2.imwrite(filename, image)

def pulse(state, element, area, input=()):
    """Make the next pulse using an element on an area, adding the input
    as a pulse source."""
    pulse1, pulse0 = state
    pulse2 = pulse1
    pulse2 = cv2.dilate(pulse2, element)
    for input_image in input:
        pulse2 = cv2.bitwise_or(pulse2, input_image)
    pulse2 = cv2.subtract(pulse2, pulse0)
    pulse2 = cv2.bitwise_and(pulse2, area)
    return pulse2, pulse1

cv2.namedWindow("PULSE", cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow("pulse_input", cv2.WINDOW_AUTOSIZE)
down_pulse = right_pulse = road_pulse = (zeros, zeros)
nor_pulse = zeros
frame_index = 0
while cv2.waitKey(1) != 27: # press escape
    t = time.time()
    pulse_input = cv2.dilate(cv2.bitwise_and(
        pulse_input_image, cv2.bitwise_not(
            cv2.dilate(nor_pulse,
                expand_element))), expand_element)
    road_pulse = pulse(road_pulse, road_pulse_element, road_image, (pulse_input, down_pulse[0], right_pulse[0]))
    right_pulse = pulse(right_pulse, right_pulse_element, gate_expanded, (road_pulse[0],))
    down_pulse = pulse(down_pulse, down_pulse_element, gate_expanded, (road_pulse[0],))
    nor_input = cv2.bitwise_and(right_pulse[0], down_pulse[0])
    nor_pulse = cv2.bitwise_or(nor_input, cv2.dilate(nor_pulse, nor_pulse_element))
    nor_pulse = cv2.bitwise_and(nor_pulse, gate_image)
    #nor_pulse = pulse(nor_pulse, nor_pulse_element, gate_expanded, (nor_input,))
    image = road_pulse[0]
    for a_pulse, strength in ((right_pulse[0], 3), (down_pulse[0], 3), (nor_pulse, 1)):
        image = cv2.bitwise_or(image, a_pulse//strength)

#    cv2.imshow("pulse_input", pulse_input)
    show_image = cv2.bitwise_or(camera_image, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    cv2.imshow("PULSE", show_image)
    print("duration", time.time() - t, "frame_index", frame_index)
    frame_index+= 1
    if OUTPUT_ANIMATION:
        output_animation(frame_index, image_path, show_image)
exit(0)
