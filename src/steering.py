import numpy as np

def compute_steering_angle(lines, width):
    if lines is None:
        return 90
    
    x_coords = []
    for line in lines:
        x1, _, x2, _ = line[0]
        x_coords.extend([x1, x2])

    avg_x = int(np.mean(x_coords))
    center = width // 2

    deviation = avg_x - center

    angle = 90 + int(deviation / center * 45)
    return angle
