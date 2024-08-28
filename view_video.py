import os
import cv2
import asyncio
import uuid
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="7il3uyauSl9F7Vif04Pb"
)

def infer_image(image_path):
    # Inference on the image (synchronous call)
    result = CLIENT.infer(image_path, model_id="traffic-and-road-signs/1")
    return result

def save_cropped_sign(frame, bbox, label, save_dir, count):
    x = int(bbox['x'] - bbox['width'] / 2)
    y = int(bbox['y'] - bbox['height'] / 2)
    w = int(bbox['width'])
    h = int(bbox['height'])

    # Ensure coordinates are within frame bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    # Crop the image
    cropped_sign = frame[y:y + h, x:x + w]
    save_path = os.path.join(save_dir, f"{label}_{count}.jpg")
    cv2.imwrite(save_path, cropped_sign)

async def process_image(image_path, save_dir):
    # Load the image
    frame = cv2.imread(image_path)

    # Save image as a temporary file for inference
    temp_image_path = f"/tmp/frame_{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Run inference on the current image (synchronous call)
    result = infer_image(temp_image_path)

    # Remove the temporary image file
    os.remove(temp_image_path)

    count = 0  # Counter for saved images

    if result and "predictions" in result:
        for prediction in result["predictions"]:
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            label = prediction['class']
            confidence = prediction['confidence']

            # Convert center coordinates (x, y) to top-left coordinates
            bbox = {
                'x': int(x - width / 2),
                'y': int(y - height / 2),
                'width': int(width),
                'height': int(height)
            }

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (bbox['x'], bbox['y']),
                          (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})",
                        (bbox['x'], bbox['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            # Save the cropped traffic sign
            save_cropped_sign(frame, bbox, label, save_dir, count)
            count += 1

    # Display the processed image
    cv2.imshow('Traffic Sign Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

async def main():
    image_path = "100km.jpeg"  # Update with your image file path
    save_dir = "cropped_signs"  # Directory to save cropped images
    os.makedirs(save_dir, exist_ok=True)

    # Process the image asynchronously
    await process_image(image_path, save_dir)

if __name__ == "__main__":
    asyncio.run(main())
