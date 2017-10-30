"""
This is the video capture for the pulse base image.
"""

from math import pi
import cv2
import time
import subprocess
import tempfile
import os
from color_classification import *

def list_video_devices():
    """List all video devices.

    Currently, this only works under Linux."""
    return [os.path.join("/dev", filename)
            for filename in os.listdir("/dev")
            if filename.startswith("video")]

def capture_image(input_device="", mirror=False):
    """Capture a video frame.

    The input device can be "" which means the default is used or
    one from list_video_devices().
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        options = []
        if mirror: options += ["mirror"]
        option_string = (["--vf-add=" + ":".join(options)] if options else [])
        try:
            called_process = subprocess.run(
                [   "mplayer", ] + option_string + [
                    "-vo", "jpeg:quality=100:outdir=" + temporary_directory,
                    "-frames", "1", "tv://" + input_device],
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE)
        except FileNotFoundError as e:
            raise FileNotFoundError(*e.args, "Install mplayer to use the camera.")
        filenames = os.listdir(temporary_directory)
        if not filenames:
            raise ValueError("Could not capture a frame from the video.",
                             called_process.stdout)
        file_path = os.path.join(temporary_directory, filenames[0])
        image = cv2.imread(file_path)
        return image

def capture_image_gray(*args, **kw):
    """Same as capture_image but returns gray picture."""
    return cv2.cvtColor(capture_image(*args, **kw), cv2.COLOR_BGR2GRAY)

def max_resolution():
    """Return the maximum resolution.

    This uses xrandr to get the resolution.
    """
    called_process = subprocess.run(["xrandr"], stdout=subprocess.PIPE)
    first_line = called_process.stdout.decode().split("\n", 1)[0]
    current = [x for x in first_line.split(",") if "current" in x][0]
    width, height = [s for s in current.split() if s.isdigit()]
    return int(width), int(height)

def max_resolution_zeros():
    res = max_resolution()
    return np.zeros(res[::-1], np.uint8)

def camera_or_argument_image(default_image="explanation.png"):
    """Image from the command line argument or capture one from the camera."""
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    elif list_video_devices():
        return capture_image()
    else:
        image_path = default
    return cv2.imread(image_path)

if __name__ == "__main__":
    print("Video devices:", list_video_devices())
    t = time.time()
    image = capture_image()
    print("max_resolution", max_resolution())
    print("capture_video", time.time() - t)
    cv2.namedWindow("Video Capture", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Video Capture", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # from https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlines
    # and  http://www.codepool.biz/opencv-line-detection.html
    # and https://stackoverflow.com/questions/19054055/python-cv2-houghlines-grid-line-detection

    element = np.array([[0,1,0], [1,0,-1], [0,-1,0]], np.uint8)
    eroded = cv2.erode(gray, element)
    cv2.namedWindow("eroded", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("eroded", eroded)
#    flag,b = cv2.threshold(eroded,0,255,cv2.THRESH_OTSU)
#    cv2.namedWindow("b", cv2.WINDOW_AUTOSIZE)
#    cv2.imshow("b", b)
    edges = cv2.Canny(eroded,150,200,3,5)
    cv2.namedWindow("edges", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("edges", edges)
    h, w = DIMENSIONS = (len(gray), len(gray[0]))
    lines = cv2.HoughLinesP(edges,1,pi/180,2, minLineLength = min(h, w)/10, maxLineGap = min(h, w)/100)
    print("lines", lines)
    for line, in lines:
        print("line", line)
        cv2.line(gray, (line[0], line[1]), (line[2], line[3]), 0)
        cv2.line(gray, (line[0] + 1, line[1] + 1), (line[2] + 1, line[3] + 1), 255)
    cv2.namedWindow("gray", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
