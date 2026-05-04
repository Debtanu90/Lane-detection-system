import cv2
import numpy as np
from config.config import get_roi_vertices

def apply_roi(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    vertices = get_roi_vertices(width, height)
    cv2.fillPoly(mask, vertices, 255)

    return cv2.bitwise_and(edges, mask)