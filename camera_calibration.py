import cv2
from camera_image import capture_image_gray, max_resolution
import numpy as np
import random

INTERACTIVE = True

CALIBRATION = "Calibration"
DIFF = "Difference"

zeros = max_resolution_zeros()
mh, mw = zeros.shape[:2]


def wait():
    if cv2.waitKey(0) == 27:
        exit(0)


def interact():
    if INTERACTIVE:
        print("Move the window and press a key.")
        wait()


cv2.namedWindow(CALIBRATION, cv2.WINDOW_AUTOSIZE)
cv2.imshow(CALIBRATION, zeros)

interact()

background = capture_image_gray()
background = background // 2


circle_radius = 30
points = (
    (circle_radius, circle_radius),
    (mw - circle_radius, circle_radius),
    (mw - circle_radius, mh - circle_radius),
    (circle_radius, mh - circle_radius),
)


images = []

EROSION_SQUARE = np.array([[1,1,1], [1,1,1], [1,1,1]], np.uint8)

cv2.namedWindow(CALIBRATION, cv2.WINDOW_AUTOSIZE)

for x, y in points:
    image = max_resolution_zeros()
    cv2.circle(image, (x, y), circle_radius, 255, -1)
    cv2.imshow(CALIBRATION, image)
    interact()
    capture = capture_image_gray()
    diff = capture // 2 + 128 - background
    ret, white = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Threshold", white)
    #images.append(white)
    # could the that connected components is faster
    count = 40
    while True:
        last_white = white
        white = cv2.erode(white, EROSION_SQUARE, iterations=count)
        white_pixel_count = cv2.countNonZero(white)
        print("white_pixel_count", white_pixel_count)
        if white_pixel_count == 0:
            white = last_white
            if count >= 4:
                count //= 4
            if count >= 2:
                count //= 2
            else:
                break # we went too far
    white_positions = cv2.findNonZero(white)
    xs, ys = cv2.split(white_positions)
    x = xs.mean()
    y = ys.mean()
    print("x, y", x, y)
    cv2.circle(capture, (int(x), int(y)), circle_radius, (255, 0, 0), 5)
    cv2.imshow(DIFF, capture)






if INTERACTIVE:
    print("Press any key to exit.")
wait()


class Calibration:

    def __init__(self, resolution="max", circle_radius=30, window_name="Calibration"):
        """Create a new Calibration window.

        Arguments:
        - resolution - a tuple (width, height) in pixels or "max" to use the
                    maximum available resolution.
        - circle_radius - the size of the circle to use for calibration.
        - window_name - the name of the window.
        """
        self._window_name = window_name
        self._resolution = resolution
        self._source_points = []
        self._destination_points = []
        self._H = None # Transformation Matrix
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self._window_name, self.get_resolution_zeros())
        cv2.moveWindow(self._window_name, 0, 0)

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
        image = self.get_resolution_zeros()
        x = int(random.random() * self._resolution[0])
        y = int(random.random() * self._resolution[1])
        cv2.circle(image, (x, y), circle_radius, 255, -1)
        cv2.imshow(self._window_name, image)
        capture = capture_image_gray()
        diff = capture // 2 + 128 - background
        ret, white = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow("Threshold", white)
        #images.append(white)
        # could the that connected components is faster
        count = 40
        while True:
            last_white = white
            white = cv2.erode(white, EROSION_SQUARE, iterations=count)
            white_pixel_count = cv2.countNonZero(white)
            print("white_pixel_count", white_pixel_count)
            if white_pixel_count == 0:
                white = last_white
                if count >= 4:
                    count //= 4
                if count >= 2:
                    count //= 2
                else:
                    break # we went too far
        white_positions = cv2.findNonZero(white)
        xs, ys = cv2.split(white_positions)
        px = xs.mean()
        py = ys.mean()
        self._source_points.append((x, y)
        self._destination_points.append((px, py))
        self._H = None

    def compute_matrix(self):
        """Compute the transformation matrix."""
        if self._H is not None:
            return
        src = np.array(self._source_points)
        dst = np.array(self._destination_points)
        assert len(src) == len(dst) >= 4, "Use record_point to get more points. You need at least 4 and have {}.".format(len(dst))
        H, x = cv2.findHomography(src, dst)
        self._H = H



if __name__ == "__main__":
    calibration = Calibration(resolution=max)
    print("Please move the window to fill the screen.")
    calibration.wait_for_key_press()
    calibration.record_points(8)
    # calibration.camera_to_projector() # input image
    # calibration.display() show an image
    calibration.wait_for_key_press()
