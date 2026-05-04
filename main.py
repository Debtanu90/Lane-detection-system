import cv2
import os
from src.lane_detection import process_frame

# =============================
# CONFIGURATION
# =============================
input_type="video"

image_path = "input/images/road.jpg"
video_path = "input/videos/road_video.mp4"

output_image_path = "output/images/output.jpg"
output_video_path = "output/videos/output.mp4"


# =============================
# IMAGE PROCESSING
# =============================

def process_image():
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Image not found")
        return 
    
    result = process_frame(image)

    cv2.imwrite(output_image_path, result)
    cv2.imshow("Lane Detection - Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================
# VIDEO PROCESSING
# =============================
def process_video():
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        print("❌ Error: Video not found")
        return
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWritter_fourcc(*"mp4v")
    out = cv2.VideoWritter(output_video_path, fourcc, fps, (width, height))

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        lane_frame = process_frame(frame)

        out.write(lane_frame)
        cv2.imshow("Lane Detection - Video", lane_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

# =============================
# MAIN FUNCTION
# =============================
if __name__ == "__main__":
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/videos", exist_ok=True)

    if input_type == "image":
        process_image()
    elif input_type == "video":
        process_video()
    else:
        print("❌ Invalid INPUT_TYPE")



