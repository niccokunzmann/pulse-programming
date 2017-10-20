import numpy as np
import cv2
import scipy.cluster.vq as vq

NUMBER_OF_COLORS = 20
NUMBER_OF_COLORS += 1 # add background color

image = cv2.imread("example.png")

flat_colors = image.reshape((-1, 3))
maximum_bright_colors = flat_colors
#def maximize_color(color):
#    m = color.sum() / 3
#    if m == 0:
#        return (255, 255, 255)
#    return [int(v * 255 / 3 // m) for v in color]
#maximum_bright_colors = np.array(list(map(maximize_color, flat_colors)), flat_colors.dtype)
print("flat_colors", set(map(tuple, flat_colors)))
whitened_colors = vq.whiten(maximum_bright_colors)
codebook, distortion = vq.kmeans(whitened_colors, NUMBER_OF_COLORS)
print("codebook", codebook, "distortion", distortion)
code, distance = vq.vq(whitened_colors, codebook)
reshaped_code = code.reshape((len(image), len(image[0])))
print("Assuming background has biggest area.")
counts = [0] * NUMBER_OF_COLORS
for cls in code:
    counts[cls] += 1
max_class = max(range(NUMBER_OF_COLORS), key=lambda i:counts[i])
classified_image = np.array(
    [[255 * (col != max_class) for col in row]
     for row in reshaped_code],
    np.uint8)

print("Eroding image to one pixel lines.")
# from http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
# from https://stackoverflow.com/questions/30045166/thinnig-of-a-binary-image-python
skel = np.zeros((len(classified_image), len(classified_image[0])), np.uint8)
temp = np.zeros((len(classified_image), len(classified_image[0])), np.uint8)
element0 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
element1 = np.array([[0,0,0], [1,1,1], [0,0,0]], np.uint8)
element2 = np.array([[0,1,0], [0,1,0], [0,1,0]], np.uint8)
element3 = np.array([[1,0,0], [0,1,0], [0,0,1]], np.uint8)
element4 = np.array([[0,0,1], [0,1,0], [1,0,0]], np.uint8)
element5 = np.array([[1,1,1], [1,1,0], [0,0,0]], np.uint8)
element6 = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)
element = np.array([[0,0,0], [0,0,1], [0,1,1]], np.uint8)

img = cv2.erode(classified_image, element, iterations=0)
done = False
for i in range(20):
    eroded = cv2.erode(img, element)
    dilated = cv2.dilate(eroded, element)
    temp = cv2.subtract(img,dilated)
    skel = cv2.bitwise_or(skel,temp)
    img = skel#eroded
    zeros = cv2.countNonZero(img)
    print(zeros)
    if zeros == 0:
        done = True
        break


print("classes", set(classified_image.flatten()))
print("classified_image", classified_image)



cv2.namedWindow("W1", cv2.WINDOW_AUTOSIZE)
cv2.imshow("W1", skel)
cv2.waitKey(0)
