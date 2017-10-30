from camera_image import capture_image, list_video_devices
import numpy as np
import cv2
import sys
import random
from scipy.optimize import minimize

if len(sys.argv) >= 2:
    image = sys.argv[1]
#elif list_video_devices():
#    image = capture_image()
else:
    image = "photos/lines_screenshot_25.10.2017.png"

if isinstance(image, str):
    image = cv2.imread(image)

captured_image = capture_image()

def find_homography_old(src, dst):
    """Return the homography matrix so that the squared distances between the
    points are minimized. There are no outliers.
    """
    assert len(src) == len(dst) >= 4
    points = [
        (np.array([x1, y1, 1]), np.array([x2, y2, 1]))
        for (x1, y1), (x2, y2) in zip(src, dst)]
    def distance_squared(p1, p2):
        return abs(p1[0] - p2[0])**2 + abs(p1[1] - p2[1])**2
    def fitness(params):
        matrix = np.array(tuple(params) + (1,))
        matrix.resize((3,3))
        return sum(distance_squared(matrix.dot(p1), p2) for p1, p2 in points)
    START = (0,1,2,3,4,5,6,7)
    result = minimize(fitness, START, options={"maxiter":100000})
    print(result)
    matrix = np.array(tuple(result.x) + (1,))
    matrix.resize((3,3))
    return matrix

def find_homography(source, destination):
    """Return the homography matrix so that the squared distances between the
    points are minimized. There are no outliers.
    """
    assert len(source) == len(destination) == 4
    rows = []
    X = 0
    Y = 1
    for i in range(4):
        src = source[i]
        dst = destination[i]

        rows.append([
            src[X],
    		src[Y],
    		1,
    		0,
    		0,
    		0,
    		-dst[X] * src[X],
    		-dst[X] * src[Y],
    		-dst[X],
        ])
        rows.append([
    		0,
		    0,
		    0,
		    src[X],
		    src[Y],
		    1,
		    -dst[Y] * src[X],
		    -dst[Y] * src[Y],
		    -dst[Y],
        ])
    rows.append([0]*9)
    svd = w, u, vt = cv2.SVDecomp(np.array(rows, float), cv2.SVD_FULL_UV)
    #print("svd", svd)
    return np.array([vt.item(8, i) for i in range(9)]).reshape(3,3)

points = [(0,0), (0, 0.5), (0,1), (0.5, 1), (1,1), (1, 0.5), (1, 0), (0.5, 0)]
ch, cw = captured_image.shape[:2]
print("ch, cw", ch, cw)
points = [(0,0), (0,ch), (cw,ch), (cw, 0)]
recognized_points = []


MOUSE_CLICK = 1
def mouseCallback(event_type, x, y, *args):
    if event_type != MOUSE_CLICK:
        return
    print("Click at", x, y, "click to go", len(points) - len(recognized_points))
    if len(recognized_points) < len(points):
        recognized_points.append((x, y))
    if len(recognized_points) >= len(points):
        # findHomography
        # getPerspectiveTransform does not work
        #H = find_homography(np.array(points), np.array(recognized_points),)#cv2.LMEDS)
        # from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(recognized_points, points, image.shape[::-1],None,None)
        #h,  w = image.shape[:2]
        #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        #H = newcameramtx
        H, x = cv2.findHomography(np.array(points), np.array(recognized_points))
        #H = find_homography(np.array(points), np.array(recognized_points))
        print("x", x, "H", H)
        image2 = cv2.warpPerspective(captured_image, H, image.shape[1::-1])
        for x1, y1, x2, y2 in ((0,0,cw,ch), (cw,0,0,ch)):
            x1_, y1_, s1_ = H.dot(np.array([x1, y1, 1]))
            x2_, y2_, s2_ = H.dot(np.array([x2, y2, 1]))
            x1_, y1_ = x1_/s1_, y1_/s1_
            x2_, y2_ = x2_/s2_, y2_/s2_
            print("x1, y1, x2, y2", (x1, y1), (x2, y2), "->", (x1_, y1_), (x2_, y2_))
            x1_, x2_, y1_, y2_ = map(int, (x1_, x2_, y1_, y2_))
            cv2.line(image, (x1_, y1_), (x2_, y2_), (0,0,0))
            cv2.line(image, (x1_+1, y1_+1), (x2_+1, y2_+1), (255,255,255))
        cv2.imshow("Video Capture", image)
        #cv2.imshow("WarpedImage", image2)
        for y, row in enumerate(image2):
            for x, pixel in enumerate(row):
                if not(pixel[0] == pixel[1] == pixel[2] == 0):
                    image[y][x] = pixel
        cv2.imshow("Mask", image)



cv2.namedWindow("Video Capture", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Video Capture", image)
cv2.setMouseCallback("Video Capture", mouseCallback)

cv2.waitKey(0)

# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
# https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#Mat%20findHomography(InputArray%20srcPoints,%20InputArray%20dstPoints,%20int%20method,%20double%20ransacReprojThreshold,%20OutputArray%20mask)
