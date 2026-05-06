import unittest
import numpy as np
import cv2

from src.preprocessing import preprocess
from src.roi import apply_roi
from src.hough import hough_lines
from src.lane_detection import process_frame

test_image_dir = "test/test_data/images/road_line_images"

class aneDetectionRealData(unittest.TestCase):
    
    def test_real_images(self):
        for file in os.listdir(test_image_dir):
            if file.endswith((".jpg", ".png")):
                path = os.path.join(test_image_dir, file)

                image = cv2.imread(path)
                self.assertIsNotNone(image, f"failed to load {file}")

                output = process_frame(image)

                # Basic checks
                self.assertIsNotNone(output)
                self.assertEqual(output.shape, image.shape)

    def test_lane_detection(self):
        for file in os.listdir(test_image_dir):
            if file.endswith((".jpg", ".png")):
                path = os.path.join(test_image_dir, file)

                image = cv2.imread(path)
                edges = preprocess(image)
                roi = apply_roi(edges)
                lines = hough_lines(roi)

                # At least some lines should be detected
                self.assertIsNotNone(lines, f"No lanes detected in {file}")

    def test_edge_density(self):
        for file in os.listdir(test_image_dir):
            if file.endswith((".jpg", ".png")):
                path = os.path.join(test_image_dir, file)

                image = cv2.imread(path)
                edges = preprocess(image)
                edge_pixels = np.sum(edges > 0)
                total_pixels = edges.size

                edge_ratio = edge_pixels / total_pixels

                # Expect at least some edges (>0.5%)
                self.assertGreater(edge_ratio, 0.005, f"Too few edges in {file}")
if __name__ == "__main__":
    unittest.main()

