import numpy as np

# Canny thresholds
canny_low = 50
canny_high = 150

# Gaussian blur
blur_kernel = (5,5)

# ROI triangle (adjust for your camera)
def get_roi_vertices(width, height):
    return np.array([[
        (int(width *0.1), height),
        (int(width *0.9), height),
        (int(width *0.5), int(height *0.6))
    ]], dtype=np.int32)