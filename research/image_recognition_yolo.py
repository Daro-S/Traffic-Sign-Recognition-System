import cv2
import logging
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from detection.traffic_sign_detection import TrafficSignDetector
from preprocess.preproces import Preprocessor
from Models.traffic_sign_recognition import TrafficSignRecognitionModel
import torch
import time
import numpy as np
from playsound import playsound  # Import playsound

class DBBox:
    def __init__(self, xmin, ymin, xmax, ymax, label):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def process_image_from_url(image_url: str, output_dir: str, detection_model_path: str, recognition_model_path: str):
    start_time = time.time()

    logger = setup_logger()

    detector = TrafficSignDetector(detection_model_path, target_class_id=11) 
    recognizer = TrafficSignRecognitionModel(recognition_model_path)
    preprocessor = Preprocessor()

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    output: list[DBBox] = []

    bboxes = detector.detect(frame)
    logger.info(f"Detected {len(bboxes)} traffic sign(s)")

    result_file_path = Path(output_dir) / 'result.txt'
    with open(result_file_path, 'w') as result_file:
        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax, _conf, _cls = bbox
            
            if _conf >= 0.30:
                detected_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                detected_path = f"{output_dir}/detected_sign_{i}.png"
                cv2.imwrite(detected_path, detected_img)

                tensor = preprocessor.preprocess(frame, (int(xmin), int(ymin), int(xmax), int(ymax)))
                preprocessed_path = f"{output_dir}/preprocessed_sign_{i}.pt"
                torch.save(tensor, preprocessed_path)

                label = recognizer.predict(tensor)
                logger.info(f"Sign {i}: Predicted Label {label}")

                result_file.write(f"Sign {i}: Detected Bounding Box {bbox}, Predicted Label {label}\n")
                
                output.append(DBBox(int(xmin), int(ymin), int(xmax), int(ymax), label))
                
                # Play the corresponding MP3 file for the predicted label
                mp3_filename = Path('descriptions_mp3') / f"{label}.mp3"
                print(f"Looking for audio file: {mp3_filename}")  # Debugging: Print the path
                if mp3_filename.exists():
                    playsound(str(mp3_filename))
                else:
                    logger.warning(f"No audio file found for label {label}")

            else:
                logger.info(f"Sign {i}: Ignored due to low confidence ({_conf:.2f})")

    logger.info("Image processing completed.")

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    render_image_with_bboxes(frame, output, output_dir)

def render_image_with_bboxes(frame, dbboxes, output_dir):
    for bbox in dbboxes:
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        label_str = str(bbox.label)
        cv2.putText(frame, label_str, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image_path = Path(output_dir) / 'output_image_with_bboxes.png'
    cv2.imwrite(str(output_image_path), frame)
    print(f"Output image saved to {output_image_path}")

def main():
    import pathlib as Path
    script_dir = Path(__file__).parent
    image_url = "https://www.portland.gov/sites/default/files/styles/facebook/public/2022/stop-sign1.jpg?itok=jOxx-iED"
    output_dir = str(script_dir / 'yolov10' / 'output')
    detection_model_path = str(script_dir / 'Models' / 'yolov10s.pt')
    recognition_model_path = str(script_dir / 'Models' / 'traffic_sign_recognition_GTSRB_model.pth')

    process_image_from_url(image_url, output_dir, detection_model_path, recognition_model_path)

if __name__ == "__main__":
    main()
