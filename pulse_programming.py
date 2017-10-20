#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.cluster.vq as vq

codebook_raw = [[0,0,0], [255, 255, 255], [255,0,0], [0,255,0], [0,0,255]]
NUMBER_OF_COLORS = len(codebook_raw)

image = cv2.imread("pathxxx.png")
DIMENSIONS = (len(image), len(image[0]))

image = cv2.GaussianBlur(image, (5, 5), 0)
flat_colors = image.reshape((-1, 3))
whitened = vq.whiten(np.concatenate((flat_colors, np.array(codebook_raw, np.uint8))))
codebook, distortion = vq.kmeans(whitened, NUMBER_OF_COLORS)
print("codebook", codebook)
all_features = whitened[:len(flat_colors)]
codebook_features = whitened[len(flat_colors):]
code, distance = vq.vq(all_features, codebook)
codebook_classification, distance = vq.vq(codebook_features, codebook)

reshaped_code = code.reshape(DIMENSIONS)
print("Assuming background has biggest area.")

background_class = codebook_classification[1]
road_source_class = codebook_classification[0]
bridge_source_class = codebook_classification[3]
pulse_source_class = codebook_classification[2]
pump_source_class = codebook_classification[4]

road_image = np.array(
    [[255 * (col == road_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)
pulse_source_image = np.array(
    [[255 * (col == pulse_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)
pump_source_image = np.array(
    [[255 * (col == pump_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)
bridge_source_image = np.array(
    [[255 * (col == bridge_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)

cv2.namedWindow("road_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("road_image", road_image)
cv2.namedWindow("pulse_source_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("pulse_source_image", pulse_source_image)
cv2.namedWindow("pump_source_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("pump_source_image", pump_source_image)
cv2.namedWindow("bridge_source_image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("bridge_source_image", bridge_source_image)

cv2.waitKey(0)
exit(0)

# pulse generation
pulse0 = zeros = np.zeros(DIMENSIONS, np.uint8)
pulse1 = np.zeros(DIMENSIONS, np.uint8)

pulse_element = np.array([[1,1,0],[1,0,0],[0,0,0]], np.uint8)
border_element = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)

def pulse(state):
    pulse1, pulse0 = state
    pulse2 = cv2.dilate(pulse1, pulse_element)
    pulse2 = cv2.subtract(pulse2, pulse0)
    pulse2 = cv2.bitwise_and(pulse2, expansion_room)
    return pulse2, pulse1

LEVEL_128 = np.full(DIMENSIONS, 128, np.uint8)
border = cv2.dilate(road_image, border_element)
border = cv2.subtract(border, road_image)
pulse1 = cv2.bitwise_and(LEVEL_128, border)
pulse1 = cv2.bitwise_and(pulse_source_image, pulse1)
cv2.namedWindow("border", cv2.WINDOW_AUTOSIZE)
cv2.imshow("border", border)
expansion_room_128 = cv2.bitwise_and(road_image, LEVEL_128)
#expansion_room = cv2.bitwise_or(expansion_room_128, expansion_room_64)
expansion_room = expansion_room_128
cv2.namedWindow("expansion_room", cv2.WINDOW_AUTOSIZE)
cv2.imshow("expansion_room", expansion_room)
state0 = state = pulse1, pulse0
cv2.namedWindow("W3", cv2.WINDOW_AUTOSIZE)
while cv2.waitKey(20) != 27: # press escape
    cv2.imshow("W3", cv2.multiply(state[0], 4))
    state = pulse(state)
    print(list(map(bin, set(list(state[0].reshape((-1)))))))
    if cv2.countNonZero(state[0]) == 0:
        state = state0
