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
        self._projection_points = []
        self._camera_points = []
        self._H = None # Transformation Matrix
        self._HI = None # Transformation Matrix inverted
        self._maximum_background_seconds = maximum_background_seconds
        self._next_background_capture = 0
        self._background = None
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self._window_name, 0, 0)
        self.fill_black()

    @property
    def height(self):
        """Height of the output window."""
        return self._resolution[1]

    @property
    def width(self):
        """Width of the output window."""
        return self._resolution[0]

    def fill_black(self):
        """Fill the window with black."""
        cv2.imshow(self._window_name, self.get_resolution_zeros())

    def record_background(self):
        """Record a background image.

        The background im age is a 7bit gray recorded image.
        """
        self.fill_black()
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
        If Space is pressed, the function resumes.
        """
        while True:
            press = cv2.waitKey(0)
            if press == 27:
                exit(0)
            if press == 32:
                return

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
        white_positions = cv2.findNonZero(white)
        xs, ys = cv2.split(white_positions)
        # we can assume there are no outliers.
        # If there were, we should use the median.
        px = xs.mean()
        py = ys.mean()
        cv2.circle(capture, (int(px), int(py)), self._circle_radius, (255, 255, 255), 2)
        cv2.imshow("Threshold", capture)
        self._projection_points.append((x, y))
        self._camera_points.append((px, py))
        self._H = None

    def compute_matrix(self):
        """Compute the transformation matrix."""
        if self._H is not None:
            return self._H
        src = np.array(self._projection_points)
        dst = np.array(self._camera_points)
        assert len(src) == len(dst) >= 4, "Use record_point() to get more points. You need at least 4 and have {}.".format(len(dst))
        H, x = cv2.findHomography(src, dst, cv2.RANSAC)
        print("x", x)
        self._H = H
        self._HI = np.linalg.inv(H)
        return self._H

    def get_points(self):
        return list(zip(self._projection_points, self._camera_points))

    def warp_projection_in_camera(self, projector_image=None):
        """Warp the projector image into the camera perspective.

        This is usally not needed.
        """
        if projector_image is None:
            projector_image = np.zeros((self.height, self.width, 3), np.uint8) + 255
        H = self.compute_matrix()
        return cv2.warpPerspective(projector_image, H, self._background.shape[1::-1])

    def area_in_camera(self, projector_image=None):
        """Show the displayed area inside the camera image.

        Parameters:
        - projector_image - the image to warp into the pespective.
                            Default: None => white image.

        This is just useful for debugging purposes.
        """
        captured_image = capture_image()
        warped_image = self.warp_projection_in_camera(projector_image)
        for y, row in enumerate(warped_image):
            for x, pixel in enumerate(row):
                if not(pixel[0] == pixel[1] == pixel[2] == 0):
                    captured_image[y][x] = pixel
        return captured_image

    def warp_camera_in_projection(self, camera_image=None):
        """Return the camera image scaled to the projection plane.

        Parameters:
        - camera_image - The image of the camera.
                         Default: None, capture an image.
        """
        if camera_image is None:
            camera_image = capture_image()
        self.compute_matrix()
        return cv2.warpPerspective(camera_image, self._HI, (self.width, self.height))

    def show_area_in_camera(self):
        """Shortcut to open a window for area_in_camera."""
        image = self.area_in_camera()
        h, w = self.height, self.width
        print("h, w", h, w)
        p00, p01, p10, p11 = self.projection_to_camera_points(
            ((0, 0), (0, h), (w, 0), (w, h)), type=int)
        cv2.line(image, p00, p11, (0,0,0), 3)
        cv2.line(image, p00, p11, (255,255,255))
        cv2.line(image, p10, p01, (0,0,0), 3)
        cv2.line(image, p10, p01, (255,255,255))
        print("p00, p01, p10, p11", p00, p01, p10, p11)
        window = "Capture Area - {}".format(self._window_name)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window, image)

    def camera_to_projection_points(self, xys, type=float):
        """Convert an array of camera points to the projection points."""
        return list(map(lambda p: tuple(map(type, p)), map(self.camera_to_projection_point, xys)))

    def camera_to_projection_point(self, xy):
        """Convert a camera point to the projection point."""
        x, y, s = self._HI.dot(np.array([xy[0], xy[1], 1]))
        return (x/s, y/s)

    def projection_to_camera_points(self, xys, type=float):
        """Convert an array of projection points to camera points."""
        return list(map(lambda p: tuple(map(type, p)), map(self.projection_to_camera_point, xys)))

    def projection_to_camera_point(self, xy):
        """Convert a projection point to camera point."""
        x, y, s = self._H.dot(np.array([xy[0], xy[1], 1]))
        return (x/s, y/s)



if __name__ == "__main__":
    window = "Calibration"
    calibration = Calibration((1024, 768), window_name=window)
    cv2.namedWindow("Threshold", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Drawing", cv2.WINDOW_AUTOSIZE)
    print("Please move the window to fill the screen and press any key.")
    calibration.wait_for_key_press()
    calibration.record_points(20)
    # calibration.camera_to_projector() # input image
    # calibration.display() show an image
    print("Matrix:", calibration.compute_matrix())
    print("Points:", calibration.get_points())
    calibration.show_area_in_camera()
    MOUSE_CLICK = 1
    def capture_mouse(event_type, x, y, *args):
        x2, y2 = calibration.camera_to_projection_point((x, y))
        print("on camera {}, {} => {}, {} on projector".format(x, y, x2, y2))
    cv2.setMouseCallback("Capture Area - {}".format(window), capture_mouse)
    def window_mouse(event_type, x, y, *args):
        x2, y2 = calibration.projection_to_camera_point((x, y))
        print("on projector {}, {} => {}, {} on camera".format(x, y, x2, y2))
    cv2.setMouseCallback(window, window_mouse)
    cv2.imshow("Drawing", calibration.warp_camera_in_projection())
    calibration.wait_for_key_press()
