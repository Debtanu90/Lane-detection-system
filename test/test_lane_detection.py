import unittest
import numpy as np
import cv2

from src.preprocessing import preprocess
from src.roi import apply_roi
from src.hough import hough_lines
from src.lane_detection import process_frame

class TestLaneDetection(unittest.TestCase):

    def setUp(self):
        # Create a dummy test image (black image with white lines)
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(self.test_image, (200, 480), (300, 300), (255, 255, 255), 5)
        cv2.line(self.test_image, (440, 480), (340, 300), (255, 255, 255), 5)

    # -----------------------------
    # Test preprocessing step
    # -----------------------------
    def test_process(self):
        edges = preprocess(self.test_image)
        self.assertIsNotNone(edges)
        self.assertEqual(len(edges.shape), 2)

    # -----------------------------
    # Test ROI masking
    # -----------------------------
    def test_roi(self):
        edges = preprocess(self.test_image)
        roi = apply_roi(edges)
        self.assertIsNotNone(roi)

    # -----------------------------
    # Test Hough line detection
    # -----------------------------
    def test_hough_lines(self):
        edges = preprocess(self.test_image)
        roi = apply_roi(edges)
        lines = hough_lines(roi)
        self.assertIsNotNone(lines)

    # -----------------------------
    # Test full pipeline
    # -----------------------------
    def test_process_frame(self):
        output = process_frame(self.test_image)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, self.test_image.shape)

if __name__ == "__main__":
    unittest.main()

