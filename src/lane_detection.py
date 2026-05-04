from src.preprocessing import preprocess
from src.roi import apply_roi
from src.hough import hough_lines
from src.draw import draw_lines
from src.steering import compute_steering_angle
import cv2

def process_frame(frame):
    edges = preprocess(frame)
    roi = apply_roi(edges)
    lines = hough_lines(roi)

    output = draw_lines(frame, lines)

    
    # Steering angle
    angle = compute_steering_angle(lines, frame.shape[1])

    # Display angle
    cv2.putText(
        output,
        f"Steering Angle: {angle}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return output