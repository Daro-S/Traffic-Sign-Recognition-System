import cv2
import logging
from pathlib import Path
from detection.traffic_sign_detection import TrafficSignDetector
from preprocess.preproces import Preprocessor
from Models.traffic_sign_recognition import TrafficSignRecognitionModel
import torch
import time

class DBBox:
    def __init__(self, xmin, ymin, xmax, ymax, label, frame):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.frame_count = frame

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for playback.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Output Video', frame)
        if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def render_video_with_bboxes(video_path, dbboxes):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    script_dir = Path(__file__).parent
    output_path = str(script_dir / 'yolov10' / 'output' / 'output_video_with_bboxes.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Frame counter
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding boxes on the current frame
        for bbox in dbboxes:
            if bbox.frame_count == frame_count:
                cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
                # Ensure label is a string before passing to cv2.putText
                label_str = str(bbox.label)
                cv2.putText(frame, label_str, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame into the file
        out.write(frame)
        
        # Display the frame (optional)
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_count += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

    play_video(output_path)

def main():
    start_time = time.time()

    script_dir = Path(__file__).parent
    video_path = str(script_dir / 'video' / 'testing_vid.mp4')
    output_dir = str(script_dir / 'yolov10' / 'output')
    detection_model_path = str(script_dir / 'Models' / 'yolov10n.pt')
    recognition_model_path = str(script_dir / 'Models' / 'traffic_sign_recognition_GTSRB_model.pth')

    logger = setup_logger()
    
    detector = TrafficSignDetector(detection_model_path, target_class_id=11) 
    recognizer = TrafficSignRecognitionModel(recognition_model_path)
    preprocessor = Preprocessor()

    cap = cv2.VideoCapture(video_path)
    output:list[DBBox] = []
    frame_count = 0

    # Open the result file in write mode
    result_file_path = str(script_dir / 'yolov10' / 'output' / 'result.txt')
    with open(result_file_path, 'w') as result_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bboxes = detector.detect(frame)
            logger.info(f"Frame {frame_count}: Detected {len(bboxes)} traffic sign(s)")

            for i, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax, _conf, _cls = bbox
                
                # Only process if the confidence level is above the threshold
                if _conf >= 0.80:
                    detected_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    detected_path = f"{output_dir}/detected_sign_{frame_count}_{i}.png"
                    cv2.imwrite(detected_path, detected_img)

                    # Preprocess the image
                    tensor = preprocessor.preprocess(frame, (int(xmin), int(ymin), int(xmax), int(ymax)))
                    preprocessed_path = f"{output_dir}/preprocessed_sign_{frame_count}_{i}.pt"
                    torch.save(tensor, preprocessed_path)

                    # Recognize the sign
                    label = recognizer.predict(tensor)
                    logger.info(f"Frame {frame_count}, Sign {i}: Predicted Label {label}")

                    # Write the detection and recognition results to the file
                    result_file.write(f"Frame {frame_count}, Sign {i}: Detected Bounding Box {bbox}, Predicted Label {label}\n")
                    
                    # Save the detection and recognition results
                    output.append(DBBox(int(xmin), int(ymin), int(xmax), int(ymax), label, frame_count))
                else:
                    logger.info(f"Frame {frame_count}, Sign {i}: Ignored due to low confidence ({_conf:.2f})")

            frame_count += 1

    cap.release()
    logger.info("Video processing completed.")

    # Calculate and print the total execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    
    render_video_with_bboxes(video_path, output)

if __name__ == "__main__":
    main()
