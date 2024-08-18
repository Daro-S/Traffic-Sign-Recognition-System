import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch

class Preprocessor:
    def __init__(self, input_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess(self, frame, bbox):
        xmin, ymin, xmax, ymax = bbox
        cropped_img = frame[ymin:ymax, xmin:xmax]
        img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        return self.transform(img).unsqueeze(0)  # Add batch dimension

def save_preprocessed_image(frame, bbox, output_path):
    preprocessor = Preprocessor()
    tensor = preprocessor.preprocess(frame, bbox)
    torch.save(tensor, output_path)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    script_dir = Path(__file__).parent
    image_path = script_dir / 'yolov10' / 'detected_sign_0_0.png'
    output_path = script_dir / 'preprocessed' / 'preprocessed_sign_0_0.pt'

    # Example usage
    frame = cv2.imread(image_path)
    bbox = (50, 50, 150, 150)  # Example bounding box
    save_preprocessed_image(frame, bbox, output_path)
