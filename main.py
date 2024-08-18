import cv2
import logging
from pathlib import Path
from detection.traffic_sign_detection import TrafficSignDetector
from preprocess.preproces import Preprocessor
from model.traffic_sign_recognition import TrafficSignRecognitionModel
import torch
import time

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def main():
    start_time = time.time()

    script_dir = Path(__file__).parent
    video_path = str(script_dir / 'video' / 'testing_vid2.mp4')
    output_dir = str(script_dir / 'yolov10')
    detection_model_path = str(script_dir / 'model' / 'yolov10n.pt')
    recognition_model_path = str(script_dir / 'model' / 'traffic_sign_recognition_GTSRB_model.pth')

    logger = setup_logger()
    
    detector = TrafficSignDetector(detection_model_path, target_class_id=11) 
    recognizer = TrafficSignRecognitionModel(recognition_model_path)
    preprocessor = Preprocessor()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Open the result file in write mode
    result_file_path = str(script_dir / 'result.txt')
    with open(result_file_path, 'w') as result_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bboxes = detector.detect(frame)
            logger.info(f"Frame {frame_count}: Detected {len(bboxes)} traffic sign")

            for i, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax, conf, cls = map(int, bbox)
                detected_img = frame[ymin:ymax, xmin:xmax]
                detected_path = f"{output_dir}/detected_sign_{frame_count}_{i}.png"
                cv2.imwrite(detected_path, detected_img)

                # Preprocess the image
                tensor = preprocessor.preprocess(frame, (xmin, ymin, xmax, ymax))
                preprocessed_path = f"{output_dir}/preprocessed_sign_{frame_count}_{i}.pt"
                torch.save(tensor, preprocessed_path)

                # Recognize the sign
                label = recognizer.predict(tensor)
                logger.info(f"Frame {frame_count}, Sign {i}: Predicted Label {label}")

                # Write the detection and recognition results to the file
                result_file.write(f"Frame {frame_count}, Sign {i}: Detected Bounding Box {bbox}, Predicted Label {label}\n")

            frame_count += 1

    cap.release()
    logger.info("Video processing completed.")

    # Calculate and print the total execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
