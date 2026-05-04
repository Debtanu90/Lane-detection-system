import  cv2
import numpy as np

def draw_lines(frame, lines):
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1,x2, y2 = line[0]
            cv2.line(line_image, (x1, y1,), (x2, y2), (0 ,255, 0), 8)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
