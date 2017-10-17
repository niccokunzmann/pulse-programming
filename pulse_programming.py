import numpy as np
import cv2
import scipy.cluster.vq as vq

NUMBER_OF_COLORS = 3

image = cv2.imread("example.png")

flat_colors = image.reshape((-1, 3))
print("flat_colors", set(map(tuple, flat_colors)))
whitened_colors = vq.whiten(flat_colors)
codebook, distortion = vq.kmeans(whitened_colors, NUMBER_OF_COLORS)
print("codebook", codebook, "distortion", distortion)
code, distance = vq.vq(whitened_colors, codebook)
reshaped_code = code.reshape((len(image), len(image[0])))
classified_image = np.array(
    [[[255 * col / (NUMBER_OF_COLORS - 1)] * 3 for col in row]
     for row in reshaped_code],
    image.dtype)

print("classes", set(classified_image.flatten()))
print("classified_image", classified_image)

cv2.namedWindow("W1", cv2.WINDOW_AUTOSIZE)
cv2.imshow("W1", image)
cv2.imshow("W1", classified_image)
cv2.waitKey(0)
