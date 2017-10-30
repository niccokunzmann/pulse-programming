"""Run the interactive pulse program.

Keys:
- Escape - Exit the program
- Space  - Update program image
- C      - Calibrate the image again

"""
import time
import cv2
from pulse_programming import PulseField
from camera_calibration import Calibration

window = "Camera Pulse Programming"
cv2.namedWindow("Threshold", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
calibration = Calibration((1024, 768), window_name=window)

def calibrate():
    calibration.record_points(20)
    calibration.show_area_in_camera()

print("Please move the window to fill the screen and press any key.")
calibration.wait_for_key_press()
calibrate()

def update_pulse_program_from_camera():
    calibration.fill_white()
    cv2.waitKey(1)
    image = calibration.warp_camera_in_projection()
    cv2.imshow("Capture", image)
    pulse_field.set_program_image(image, blue_threshold=0.57)

pulse_field = PulseField()
#pulse_field.DELATION_ITERATIONS = 4
#pulse_field.EROSION_ITERATIONS = 3
update_pulse_program_from_camera()
while True:
    key = cv2.waitKey(1)
    if key == 27: # Escape
        exit(0)
    elif key == 32: # Space
        update_pulse_program_from_camera()
    elif key == ord("c"): # Calibrate
        calibrate()
    t = time.time()
    pulse_field.pulse()
    print("duration:", time.time() - t)
    cv2.imshow(window, pulse_field.get_pulse_gray())
