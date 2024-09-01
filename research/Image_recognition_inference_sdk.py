import cv2
import logging
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import time
import torch
import numpy as np
from playsound import playsound
from inference_sdk import InferenceHTTPClient
from preprocess.preproces import Preprocessor
from Models.traffic_sign_recognition import TrafficSignRecognitionModel
from label.label import classes_labels

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

def infer_image(image):
    # Convert the image to a temporary path if needed (this can be optimized)
    temp_image_path = f"/tmp/temp_image.jpg"
    cv2.imwrite(temp_image_path, image)

    # Inference using InferenceHTTPClient
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="7il3uyauSl9F7Vif04Pb"
    )
    result = CLIENT.infer(temp_image_path, model_id="traffic-and-road-signs/1")
    
    return result

def process_image_from_url(image_url: str, output_dir: str, recognition_model_path: str):
    start_time = time.time()
    logger = setup_logger()

    recognizer = TrafficSignRecognitionModel(recognition_model_path)
    preprocessor = Preprocessor()

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = infer_image(frame)

    output: list[DBBox] = []

    if result and "predictions" in result:
        logger.info(f"Detected {len(result['predictions'])} traffic sign(s)")
        
        result_file_path = Path(output_dir) / 'result.txt'
        with open(result_file_path, 'w') as result_file:
            for i, prediction in enumerate(result['predictions']):
                xmin = int(prediction['x'] - prediction['width'] / 2)
                ymin = int(prediction['y'] - prediction['height'] / 2)
                xmax = int(prediction['x'] + prediction['width'] / 2)
                ymax = int(prediction['y'] + prediction['height'] / 2)
                confidence = prediction['confidence']

                if confidence >= 0.30:
                    # Crop the detected traffic sign
                    cropped_img = frame[ymin:ymax, xmin:xmax]
                    detected_path = f"{output_dir}/detected_sign_{i}.png"
                    cv2.imwrite(detected_path, cropped_img)

                    # Prepare the bounding box in the format expected by the preprocessor
                    bbox = (xmin, ymin, xmax, ymax)

                    # Preprocess the cropped image for the recognition model
                    tensor = preprocessor.preprocess(frame, bbox)  # Pass the bbox argument
                    preprocessed_path = f"{output_dir}/preprocessed_sign_{i}.pt"
                    torch.save(tensor, preprocessed_path)

                    # Predict the label using the custom model
                    label = recognizer.predict(tensor)
                    logger.info(f"Sign {i}: Predicted Label {label}")

                    result_file.write(f"Sign {i}: Detected Bounding Box ({xmin}, {ymin}, {xmax}, {ymax}), Predicted Label {label}\n")

                    output.append(DBBox(xmin, ymin, xmax, ymax, label))
                    
                    # # Play the corresponding MP3 file for the predicted label
                    # mp3_filename = Path('descriptions_mp3') / f"{label}.mp3"
                    # if mp3_filename.exists():
                    #     playsound(str(mp3_filename))
                    # else:
                    #     logger.warning(f"No audio file found for label {label}")
                else:
                    logger.info(f"Sign {i}: Ignored due to low confidence ({confidence:.2f})")
    else:
        logger.info("No traffic signs detected.")

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
    script_dir = Path(__file__).parent
    image_url = "https://c8.alamy.com/comp/HX2H5M/freight-transport-traffic-is-prohibited-road-sign-isolated-on-white-HX2H5M.jpg"
    output_dir = str(script_dir / 'inference_sdk' / 'output')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    recognition_model_path = str(script_dir / 'Models' / 'traffic_sign_recognition_GTSRB_model.pth')

    process_image_from_url(image_url, output_dir, recognition_model_path)

if __name__ == "__main__":
    main()
