from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
import cv2
import torch
from detection.traffic_sign_detection import TrafficSignDetector
from preprocess.preproces import Preprocessor
from Models.traffic_sign_recognition import TrafficSignRecognitionModel
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Set up the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the DBBox class
class DBBox:
    def __init__(self, xmin, ymin, xmax, ymax, label, frame):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.frame_count = frame

# Initialize your models
detection_model_path = str(Path(__file__).parent / 'Models' / 'yolov10n.pt')
recognition_model_path = str(Path(__file__).parent / 'Models' / 'traffic_sign_recognition_GTSRB_model.pth')
detector = TrafficSignDetector(detection_model_path, target_class_id=11)
recognizer = TrafficSignRecognitionModel(recognition_model_path)
preprocessor = Preprocessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Debugging: Add logging to verify the file type check
            print(f"File received: {filename}")

            # Correctly identify image or video and process accordingly
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print("Processing as image")
                results = process_image(file_path)
                return render_template('result.html', results=results, file_path=file_path)
            elif filename.lower().endswith('.mp4'):
                print("Processing as video")
                output_video_path = process_video(file_path)
                return render_template('result.html', file_path=output_video_path)
            else:
                flash('Unsupported file type')
                return redirect(request.url)
    
    return render_template('index.html')

def process_image(image_path):
    img = cv2.imread(image_path)
    bboxes = detector.detect(img)
    results = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, _conf, _cls = bbox
       
        if _conf >= 0.80:  # Adjust this threshold if needed
            tensor = preprocessor.preprocess(img, (int(xmin), int(ymin), int(xmax), int(ymax)))
            label = recognizer.predict(tensor)
            results.append({
                'label': label,
                'bbox': (int(xmin), int(ymin), int(xmax), int(ymax)),
                'confidence': _conf  # Add confidence to results
            })
    return results


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output = []
    frame_count = 0

    script_dir = Path(__file__).parent
    output_dir = script_dir / 'static' / 'uploads'
    output_video_path = str(output_dir / 'output_video_with_bboxes.mp4')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        bboxes = detector.detect(frame)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, _conf, _cls = bbox
            if _conf >= 0.10:
                tensor = preprocessor.preprocess(frame, (int(xmin), int(ymin), int(xmax), int(ymax)))
                label = recognizer.predict(tensor)
                label_str = str(label)
                output.append(DBBox(int(xmin), int(ymin), int(xmax), int(ymax), label_str, frame_count))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    return output_video_path

if __name__ == '__main__':
    app.run(debug=True)
