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
    [[col == max_class for col in row]
     for row in reshaped_code],
    np.uint8)

print("Eroding image to one pixel lines.")


print("classes", set(classified_image.flatten()))
print("classified_image", classified_image)



cv2.namedWindow("W1", cv2.WINDOW_AUTOSIZE)
cv2.imshow("W1", image)
cv2.imshow("W1", classified_image)
cv2.waitKey(0)
