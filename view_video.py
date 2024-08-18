import logging
from pathlib import Path
from detection.traffic_sign_detection import TrafficSignDetector

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def main():
    script_dir = Path(__file__).parent
    video_path = str(script_dir / 'video' / 'testing_vid.mp4')
    output_dir = Path(script_dir / 'yolov8')
    detection_model_path = str(script_dir / 'model' / 'yolov8n.pt')
    recognition_model_path = str(script_dir / 'model' / 'traffic_sign_recognition_GTSRB_model.pth')

    logger = setup_logger()

    # Initialize the TrafficSignDetector with the YOLO model path
    detector = TrafficSignDetector(detection_model_path, target_class_id=11)

    # Perform detection on the video and display it
    detector.detect_and_display(video_path, conf=0.2, save=True)

if __name__ == "__main__":
    main()
