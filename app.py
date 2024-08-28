from flask import Flask, request, jsonify, render_template, redirect, url_for
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

app = Flask(__name__)

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
    temp_image_path = f"/tmp/temp_image.jpg"
    cv2.imwrite(temp_image_path, image)

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

    output = []
    label = None
    label_name = None

    if result and "predictions" in result:
        logger.info(f"Detected {len(result['predictions'])} traffic sign(s)")
        
        for i, prediction in enumerate(result['predictions']):
            xmin = int(prediction['x'] - prediction['width'] / 2)
            ymin = int(prediction['y'] - prediction['height'] / 2)
            xmax = int(prediction['x'] + prediction['width'] / 2)
            ymax = int(prediction['y'] + prediction['height'] / 2)
            confidence = prediction['confidence']

            if confidence >= 0.30:
                cropped_img = frame[ymin:ymax, xmin:xmax]
                detected_path = f"{output_dir}/detected_sign_{i}.png"
                cv2.imwrite(detected_path, cropped_img)

                bbox = (xmin, ymin, xmax, ymax)

                tensor = preprocessor.preprocess(frame, bbox)
                preprocessed_path = f"{output_dir}/preprocessed_sign_{i}.pt"
                torch.save(tensor, preprocessed_path)

                label = recognizer.predict(tensor)
                label_name = classes_labels.get(label, "Unknown")
                logger.info(f"Sign {i}: Predicted Label {label_name} (Class {label})")

                output.append(DBBox(xmin, ymin, xmax, ymax, label))
                
                mp3_filename = Path('descriptions_mp3') / f"{label}.mp3"
                if mp3_filename.exists():
                    playsound(str(mp3_filename))
                else:
                    logger.warning(f"No audio file found for label {label}")
            else:
                logger.info(f"Sign {i}: Ignored due to low confidence ({confidence:.2f})")
    else:
        logger.info("No traffic signs detected.")

    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    render_image_with_bboxes(frame, output, output_dir)

    return label, label_name  # Return the predicted label and its name


def render_image_with_bboxes(frame, dbboxes, output_dir):
    for bbox in dbboxes:
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        label_str = str(bbox.label)
        cv2.putText(frame, label_str, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image_path = Path(output_dir) / 'output_image_with_bboxes.png'
    cv2.imwrite(str(output_image_path), frame)
    print(f"Output image saved to {output_image_path}")

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = None
    predicted_label_name = None
    image_url = None

    if request.method == 'POST':
        image_url = request.form['image_url']
        output_dir = 'output'
        recognition_model_path = 'Models/traffic_sign_recognition_GTSRB_model.pth'
        
        predicted_label, predicted_label_name = process_image_from_url(image_url, output_dir, recognition_model_path)
        
    return render_template('index.html', image_url=image_url, predicted_label=predicted_label, predicted_label_name=predicted_label_name)


if __name__ == "__main__":
    app.run(debug=True)
