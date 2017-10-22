
flat_colors = image.reshape((-1, 3))
whitened = vq.whiten(np.concatenate((flat_colors, np.array(codebook_raw, np.uint8))))
all_features = whitened[:len(flat_colors)]
codebook_whitened = whitened[len(flat_colors):]
codebook_whitened = len(codebook_whitened)
codebook, distortion = vq.kmeans(whitened, codebook_whitened)
print("codebook", codebook)
code, distance = vq.vq(all_features, codebook)

reshaped_code = code.reshape(DIMENSIONS)
print("Assuming background has biggest area.")

color_usage = [0] * NUMBER_OF_COLORS
for cls in code:
    color_usage[cls] += 1
background_class = max(range(NUMBER_OF_COLORS), key=lambda i: color_usage[i])
print("color_usage", color_usage, "background_class", background_class)
colorful_class = max(range(NUMBER_OF_COLORS), key=lambda i: how_colored_is_this(codebook[i]))
print("colorful_class", colorful_class)
assert colorful_class != background_class, "The background must not be colored."
gate_source_class = colorful_class
road_source_class = (set(range(NUMBER_OF_COLORS)) - set((colorful_class, background_class))).pop()

road_image = np.array(
    [[255 * (col == road_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)
gate_image = np.array(
    [[255 * (col == gate_source_class) for col in row]
     for row in reshaped_code],
    np.uint8)
