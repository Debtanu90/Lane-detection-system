import cv2
import numpy as np

def hough_lines(edges):
    return cv2.HoughLinesP(
        edges,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=140,
        maxLineGap=5
    )