import cv2
from config.config import canny_low,canny_high,blur_kernel 

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    return edges
