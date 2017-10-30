import cv2
from camera_image import *
import numpy as np
import random
import time

EROSION_ELEMENT = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)

class Calibration:

    def __init__(self, resolution="max", circle_radius=30, window_name="Calibration",
                 maximum_background_seconds=10):
        """Create a new Calibration window.

        Arguments:
        - resolution - a tuple (width, height) in pixels or "max" to use the
                    maximum available resolution.
        - circle_radius - the size of the circle to use for calibration.
        - window_name - the name of the window.
        - maximum_background_seconds - seconds until a new background image
                                       should be recorced
        """
        self._window_name = window_name
        if resolution == "max":
            resolution = max_resolution()
        self._resolution = resolution
        self._circle_radius = circle_radius
        self._source_points = []
        self._destination_points = []
        self._H = None # Transformation Matrix
        self._maximum_background_seconds = maximum_background_seconds
        self._next_background_capture = 0
        self._background = None
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self._window_name, 0, 0)
        self.fill_black()

    def fill_black(self):
        """Fill the window with black."""
        cv2.imshow(self._window_name, self.get_resolution_zeros())

    def record_background(self):
        """Record a background image.

        The background im age is a 7bit gray recorded image.
        """
        self._background = capture_image_gray()
        self._next_background_capture = time.time() + self._maximum_background_seconds

    def provide_current_background(self):
        """Make sure the calibration process works on the current background."""
        if time.time() > self._next_background_capture:
            self.record_background()

    @staticmethod
    def wait_for_key_press():
        """Wait for a key to be pressed.

        If Escape is pressed, a SystemExit is raised.
        """
        if cv2.waitKey(0) == 27:
            exit(0)

    def record_points(self, number_of_points, interactive=False):
        """Record a number of points during the calibration process.

        Parameters:
        - number_of_points - integer of points to record.
        - interactive - bool whether to wait for a key press before a record.
        """
        for i in range(number_of_points):
            if interactive:
                self.wait_for_key_press()
            self.record_point()

    def get_resolution_zeros(self):
        """Return an array of zero values uint8 in the resolution."""
        return np.zeros(self._resolution[::-1])

    def record_point(self):
        """Display a point and record it with the camera."""
        self.provide_current_background()
        image = self.get_resolution_zeros()
        x = int( self._circle_radius + random.random() * (self._resolution[0] -  self._circle_radius * 2))
        y = int( self._circle_radius + random.random() * (self._resolution[1] -  self._circle_radius * 2))
        cv2.circle(image, (x, y), self._circle_radius, 255, -1)
        cv2.imshow(self._window_name, image)
        cv2.waitKey(1)
        capture = capture_image_gray()
        greater = cv2.compare(capture, self._background, cv2.CMP_GT)
        diff = cv2.bitwise_and(greater, capture)
        ret, white = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # TODO: evaluate: could the that connected components is faster
        count = 40
        while True:
            last_white = white
            white = cv2.erode(white, EROSION_ELEMENT, iterations=count)
            white_pixel_count = cv2.countNonZero(white)
            if white_pixel_count == 0:
                white = last_white
                if count >= 4:
                    count //= 4
                elif count >= 2:
                    count //= 2
                else:
                    break # we went too far
        cv2.imshow("Threshold", white)
        white_positions = cv2.findNonZero(white)
        xs, ys = cv2.split(white_positions)
        px = xs.mean()
        py = ys.mean()
        self._source_points.append((x, y))
        self._destination_points.append((px, py))
        self._H = None

    def compute_matrix(self):
        """Compute the transformation matrix."""
        if self._H is not None:
            return self._H
        src = np.array(self._source_points)
        dst = np.array(self._destination_points)
        assert len(src) == len(dst) >= 4, "Use record_point() to get more points. You need at least 4 and have {}.".format(len(dst))
        H, x = cv2.findHomography(src, dst, cv2.RANSAC)
        self._H = H
        return self._H

    def get_points(self):
        return list(zip(self._source_points, self._destination_points))

    def area_in_camera(self, image=np.array([[[255,255,255]]], np.uint8)):
        """Show the displayed area inside the camera image.

        Parameters:
        - image - the image to warp into the pespective.

        This is just useful for debugging purposes.
        """
        captured_image = capture_image()
        H = self.compute_matrix()
        warped_image = cv2.warpPerspective(image, H, captured_image.shape[:2])
        for y, row in enumerate(warped_image):
            for x, pixel in enumerate(row):
                if not(pixel[0] == pixel[1] == pixel[2] == 0):
                    captured_image[y][x] = pixel
        return captured_image

    def show_area_in_camera(self):
        """Shortcut to open a window for area_in_camera."""
        image = self.area_in_camera()
        window = "Capture Area - {}".format(self._window_name)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, image)

if __name__ == "__main__":
    calibration = Calibration((1024, 768))
    cv2.namedWindow("Threshold", cv2.WINDOW_AUTOSIZE)
    print("Please move the window to fill the screen and press any key.")
    calibration.wait_for_key_press()
    calibration.record_points(20)
    # calibration.camera_to_projector() # input image
    # calibration.display() show an image
    print("Matrix:", calibration.compute_matrix())
    print("Points:", calibration.get_points())
    calibration.show_area_in_camera()
    calibration.wait_for_key_press()
