
def rgb_likelyhood(color):
    """Return deviation from red, green, blue"""
    color = list(map(float, color))
    return (color[1] - color[2])**2, (color[0] - color[2])**2, (color[0] - color[1])**2

def how_colored_is_this(color):
    r, g, b = rgb_likelyhood(color)
    return g + b + r

NORM = 256 * 3

m = 0
def how_colored_is_this_normalized(c):
    global m
    r = how_colored_is_this(c) // NORM
    if r > m:
        print(r)
        m = r
    return r
