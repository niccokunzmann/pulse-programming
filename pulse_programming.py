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


class PulseField:
    """This field computes the pulse.

    Attributes:
    - DELATION_ITERATIONS - expasion at the beginning of the program to reduce noise
    - EROSION_ITERATIONS - erosion at the beginning of the program to reduce noise
    - road_pulse_element - pulse walking along the roads
    - right_pulse_element - pulse walking through the gate to the right
    - down_pulse_element - pulse walking down through the gate
    - and_pulse_element - pulse walking through the gate when down and right pulse join
    - nand_element - expansion to find NAND positions
    - expand_element - expansion of the cells to the neighbors
    - gate_expand_element - expansion of the gate to overlap roads with right and down pulse
    """

    DELATION_ITERATIONS = 3
    EROSION_ITERATIONS = 2

    road_pulse_element = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    right_pulse_element = np.array([[0,0,0],[1,0,0],[0,0,0]], np.uint8)
    down_pulse_element = np.array([[0,1,0],[0,0,0],[0,0,0]], np.uint8)
    and_pulse_element = np.array([[1,0,0],[0,0,0],[0,0,0]], np.uint8)
    nand_element = np.array([[0,0,1],[0,0,1],[1,1,1]], np.uint8)
    expand_element = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
    gate_expand_element = np.array([[1,1,1],[1,1,0],[1,0,0]], np.uint8)

    def __init__(self, program_image):
        """Create a new pulse working on an image."""
        self.set_program_image(program_image)
        self.reset_pulse()

    def set_program_image(self, program_image, debug=True):
        """Set the program image.

        The background should be white, the gates blue and the roads black.

        Parameters:
        - program_image - BGR image.
        - debug - whether to open windows with debug information.
        """
        self._program_image = program_image
        self._parse_program_image(debug)
        self._initialize_pulse_generation()

    def _parse_program_image(self, debug):
        """Parse the program image to set the road and gates."""
        program_image = self._program_image
        gate_image = get_blue_recognition(program_image)
        foreground_image = get_foreground_from_BGR(program_image)
        road_image = cv2.bitwise_and(
            cv2.bitwise_not(foreground_image),
            cv2.bitwise_not(gate_image))

        gate_image = cv2.erode(gate_image, self.expand_element, iterations=self.EROSION_ITERATIONS)
        gate_image = cv2.dilate(gate_image, self.expand_element, iterations=self.DELATION_ITERATIONS)
        road_image = cv2.dilate(road_image, self.expand_element, iterations=self.DELATION_ITERATIONS)
        road_image = cv2.erode(road_image, self.expand_element, iterations=self.EROSION_ITERATIONS)

        road_image = cv2.bitwise_and(road_image, cv2.bitwise_not(gate_image))

        if debug:
            cv2.namedWindow("road_image", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("road_image", road_image)
            cv2.namedWindow("gate_image", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("gate_image", gate_image)
        self._gate_image = gate_image
        self._road_image = road_image

    def _initialize_pulse_generation(self):
        """Based on roads and gates, initialize the pulse generation."""
        gate_image = self._gate_image
        self._pulse_input_image = cv2.subtract(
            gate_image,
            cv2.dilate(gate_image, self.nand_element))
        self._gate_expanded = cv2.dilate(gate_image, self.gate_expand_element)

    def reset_pulse(self):
        """Set the pulse state to the beginnig of the program."""
        zeros = np.zeros(self._road_image.shape()[:2], np.uint8)
        self._down_pulse = self._right_pulse = self._road_pulse = (zeros, zeros)
        self._and_pulse = zeros

    def pulse(self):
        """Do one step of pulsing."""
        pulse_input = cv2.dilate(cv2.bitwise_and(
            self._pulse_input_image, cv2.bitwise_not(
                cv2.dilate(self._and_pulse,
                    expand_element))), expand_element)
        self._road_pulse = pulse(
            self._road_pulse, self.road_pulse_element, self._road_image,
            (pulse_input, self._down_pulse[0], self._right_pulse[0]))
        self._right_pulse = pulse(
            self._right_pulse, self.right_pulse_element, self._gate_expanded,
            (self._road_pulse[0],))
        self._down_pulse = pulse(
            self._down_pulse, self.down_pulse_element, self._gate_expanded,
            (self._road_pulse[0],))
        # and pulse
        and_input = cv2.bitwise_and(right_pulse[0], down_pulse[0])
        and_pulse = cv2.bitwise_or(
            self._and_input, cv2.dilate(self._and_pulse, self.and_pulse_element))
        self._and_pulse = cv2.bitwise_and(and_pulse, self._gate_image)

    def get_pulse_gray(self):
        """Return the pulse image of the pulse."""
        image = self._road_pulse[0]
        pulses = (
            (self._right_pulse[0], 3),
            (self._down_pulse[0], 3),
            (self._and_pulse, 1)
        )
        for a_pulse, strength in pulses:
            image = cv2.bitwise_or(image, a_pulse//strength)
        return image

    def get_pulse_colored(self):
        """Return the pulse image colored."""
        gray = self.get_image_gray()
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def get_pulse_on_program(self):
        """Return the pulse image rendered on the program image"""
        colored = self.get_image_colored()
        return cv2.bitwise_or(self._program_image, colored)



if __name__ == "__main__":
    image_path = (sys.argv[1] if len(sys.argv) > 1 else "explanation.png")
    image = camera_image = cv2.imread(image_path)
    cv2.namedWindow("PULSE", cv2.WINDOW_AUTOSIZE)
    pulse_field = PulseField(image)
    while cv2.waitKey(1) != 27:
        t = time.time()
        pulse_field.pulse()
        print("duration:", time.time() - t)
        cv2.imshow("PULSE", pulse_field.get_pulse_on_program())
